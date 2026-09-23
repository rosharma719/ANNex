#!/usr/bin/env python3
"""Free local BEIR benchmark: PLAID+ColBERTv2 vs exact/vectordb MiniLM."""
import argparse, json, math, shutil, subprocess, time, urllib.error, urllib.request
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from colbert_config import MODEL_ID as COLBERT_MODEL_ID, cache_config, load as load_colbert
from data import load_slice, write_slice_manifest
from embeddings import cached_fixed, cached_ragged
from env import load_env
from provenance import write_report
from significance import paired_bootstrap

load_env()

def http(base, route, body=None):
    data=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(base+route,data,{"content-type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=600) as response:return json.load(response)
    except urllib.error.HTTPError as error:
        detail=error.read().decode(errors="replace")
        raise RuntimeError(f"{route} returned HTTP {error.code}: {detail}") from error

def score(run,qrels,k=10):
    ndcg=[];recall=[]
    for qid,relevant in qrels.items():
        ranked=run.get(qid,[])[:k];gains=[relevant.get(doc,0) for doc in ranked]
        dcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(gains));ideal=sorted(relevant.values(),reverse=True)[:k]
        idcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal));ndcg.append(dcg/idcg if idcg else 0)
        wanted={doc for doc,g in relevant.items() if g>0};recall.append(len(wanted.intersection(ranked))/len(wanted))
    return {f"ndcg@{k}":float(np.mean(ndcg)),f"recall@{k}":float(np.mean(recall))}

def colbert_encode(model,texts,is_query,batch):
    values=model.encode(texts,is_query=is_query,batch_size=batch,show_progress_bar=True)
    return [np.asarray(value,dtype=np.float32).tolist() for value in values]

def size(path):return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",default="beir/nfcorpus/test");p.add_argument("--output",type=Path,default=Path("benchmark/results/nfcorpus"));p.add_argument("--report-dir",type=Path,default=Path("benchmark/reports"));p.add_argument("--cache-dir",type=Path,default=Path("benchmark/cache"));p.add_argument("--refresh-cache",action="store_true");p.add_argument("--limit-docs",type=int);p.add_argument("--limit-queries",type=int);p.add_argument("--sampling",choices=["prefix","qrels"],default="prefix");p.add_argument("--sample-seed",type=int,default=13);p.add_argument("--centroids",type=int,default=256);p.add_argument("--probes",type=int,default=8);p.add_argument("--candidates",type=int,default=100);p.add_argument("--dense-hnsw-m",type=int,default=16);p.add_argument("--dense-ef-search",type=int,default=256);p.add_argument("--batch-size",type=int,default=32);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    docs,queries,qrels=load_slice(a.dataset,a.limit_docs,a.limit_queries,a.sampling,a.sample_seed);write_slice_manifest(a.output/"slice.json",a.dataset,a.sampling,a.sample_seed,docs,queries);texts=[(getattr(d,"title","")+" "+d.text).strip() for d in docs]

    colbert=None
    def encode_colbert(values,is_query):
        nonlocal colbert
        if colbert is None: colbert=load_colbert()
        return colbert_encode(colbert,values,is_query,a.batch_size)
    print("Loading or encoding ColBERTv2 documents")
    multi_docs,colbert_doc_cache=cached_ragged(a.cache_dir,COLBERT_MODEL_ID,"document",[d.doc_id for d in docs],texts,lambda:encode_colbert(texts,False),a.refresh_cache,cache_config("document"));dimension=multi_docs[0].shape[1]
    lengths=np.asarray([len(document) for document in multi_docs],dtype=np.int64);offsets=np.concatenate(([0],np.cumsum(lengths)));rng=np.random.default_rng(13);n=min(int(offsets[-1]),max(a.centroids*50,a.centroids));chosen=rng.choice(int(offsets[-1]),n,replace=False);document_indices=np.searchsorted(offsets[1:],chosen,side="right");samples=[np.asarray(multi_docs[document][token-offsets[document]],dtype=np.float32).tolist() for document,token in zip(document_indices,chosen)]
    plaid_path=a.output/"plaid";shutil.rmtree(plaid_path,ignore_errors=True);root=Path(__file__).resolve().parents[3]
    plaid_bin=root/"target/release/annex-multivector"
    command=[str(plaid_bin),"--dimension",str(dimension),"--centroids",str(a.centroids),"--probes",str(a.probes),"--path",str(plaid_path),"--listen","127.0.0.1:18080"]
    server=subprocess.Popen(command,cwd=root,stdout=subprocess.DEVNULL);base="http://127.0.0.1:18080"
    try:
        for _ in range(300):
            try:http(base,"/healthz");break
            except Exception:time.sleep(1)
        else:raise RuntimeError("PLAID server did not start")
        http(base,"/v1/train",{"vectors":samples,"iterations":20})
        for start in range(0,len(docs),100):http(base,"/v1/vectors/upsert",{"documents":[{"id":d.doc_id,"vectors":np.asarray(v).tolist()} for d,v in zip(docs[start:start+100],multi_docs[start:start+100])]})
        query_texts=[q.text for q in queries];multi_queries,colbert_query_cache=cached_ragged(a.cache_dir,COLBERT_MODEL_ID,"query",[q.query_id for q in queries],query_texts,lambda:encode_colbert(query_texts,True),a.refresh_cache,cache_config("query"));plaid_run={};plaid_times=[]
        for q,v in zip(queries,multi_queries):
            started=time.perf_counter();result=http(base,"/v1/query",{"vectors":np.asarray(v).tolist(),"top_k":100,"candidates":a.candidates});plaid_times.append(time.perf_counter()-started);plaid_run[q.query_id]=[x["id"] for x in result["matches"]]
        plaid_stats=http(base,"/v1/stats")
    finally:server.terminate();server.wait(timeout=30)

    dense=None
    def encode_dense(values):
        nonlocal dense
        if dense is None: dense=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return dense.encode(values,batch_size=a.batch_size,normalize_embeddings=True,show_progress_bar=True)
    print("Loading or encoding MiniLM documents and indexing vectordb")
    dense_docs,minilm_doc_cache=cached_fixed(a.cache_dir,"sentence-transformers/all-MiniLM-L6-v2","document",[d.doc_id for d in docs],texts,lambda:encode_dense(texts),True,a.refresh_cache);dense_path=a.output/"vectordb";shutil.rmtree(dense_path,ignore_errors=True);dense_bin=root/"target/release/dense-server";dense_server=subprocess.Popen([str(dense_bin),"--dimension",str(dense_docs.shape[1]),"--path",str(dense_path),"--listen","127.0.0.1:18081"],cwd=root,stdout=subprocess.DEVNULL);dense_base="http://127.0.0.1:18081"
    try:
        for _ in range(60):
            try:http(dense_base,"/healthz");break
            except Exception:time.sleep(1)
        else:raise RuntimeError("dense baseline server did not start")
        for start in range(0,len(docs),256):http(dense_base,"/v1/vectors/upsert",{"documents":[{"id":docs[i].doc_id,"vector":dense_docs[i].tolist()} for i in range(start,min(start+256,len(docs)))]})
        build_started=time.perf_counter();http(dense_base,"/v1/index",{"m":a.dense_hnsw_m,"ef_construct":a.dense_ef_search});dense_build_seconds=time.perf_counter()-build_started
        query_texts=[q.text for q in queries];dense_queries,minilm_query_cache=cached_fixed(a.cache_dir,"sentence-transformers/all-MiniLM-L6-v2","query",[q.query_id for q in queries],query_texts,lambda:encode_dense(query_texts),True,a.refresh_cache);exact_dense_run={};vectordb_run={};exact_dense_times=[];vectordb_times=[]
        for q,v in zip(queries,dense_queries):
            started=time.perf_counter();result=http(dense_base,"/v1/query",{"vector":v.tolist(),"top_k":100,"backend":"exact"});exact_dense_times.append(time.perf_counter()-started);exact_dense_run[q.query_id]=[x["id"] for x in result["matches"]]
            started=time.perf_counter();result=http(dense_base,"/v1/query",{"vector":v.tolist(),"top_k":100,"backend":"hnsw","ef_search":a.dense_ef_search});vectordb_times.append(time.perf_counter()-started);vectordb_run[q.query_id]=[x["id"] for x in result["matches"]]
        dense_stats=http(dense_base,"/v1/stats")
    finally:dense_server.terminate();dense_server.wait(timeout=30)
    report={"dataset":a.dataset,"sampling":a.sampling,"sample_seed":a.sample_seed,"documents":len(docs),"queries":len(queries),"embedding_cache":{"colbert_documents":colbert_doc_cache,"colbert_queries":colbert_query_cache,"minilm_documents":minilm_doc_cache,"minilm_queries":minilm_query_cache},"systems":{"muvera_colbertv2":{**score(plaid_run,qrels),"p50_ms":float(np.percentile(plaid_times,50)*1000),"p95_ms":float(np.percentile(plaid_times,95)*1000),"storage_bytes":size(plaid_path),"index_stats":plaid_stats},"exact_minilm":{**score(exact_dense_run,qrels),"p50_ms":float(np.percentile(exact_dense_times,50)*1000),"p95_ms":float(np.percentile(exact_dense_times,95)*1000),"storage_bytes":int(dense_docs.nbytes)},"vectordb_hnsw_minilm":{**score(vectordb_run,qrels),"p50_ms":float(np.percentile(vectordb_times,50)*1000),"p95_ms":float(np.percentile(vectordb_times,95)*1000),"storage_bytes":size(dense_path),"build_seconds":dense_build_seconds,"index_stats":dense_stats,"m":a.dense_hnsw_m,"ef_search":a.dense_ef_search}},"comparisons":{"muvera_colbertv2_vs_exact_minilm":paired_bootstrap(exact_dense_run,plaid_run,qrels,seed=a.sample_seed)}};path=write_report(a.report_dir,"run",report);print(path);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
