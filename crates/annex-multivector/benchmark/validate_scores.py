#!/usr/bin/env python3
"""Compare reference, Rust-uncompressed, and stored-compressed MaxSim scores."""
import argparse, json, subprocess, time
from pathlib import Path
import ir_datasets, numpy as np
from pylate import rank
from colbert_config import MODEL_ID, cache_config, load as load_colbert
from env import load_env
from embeddings import cached_ragged
from run import colbert_encode, http
from provenance import write_report

load_env()

def numpy_maxsim(query, document):
    q=np.asarray(query,dtype=np.float32);d=np.asarray(document,dtype=np.float32)
    q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12);d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-12)
    return float((q@d.T).max(axis=1).sum())

def main():
    p=argparse.ArgumentParser();p.add_argument("--index",type=Path,default=Path("benchmark/results/nfcorpus-muvera-binary/plaid"));p.add_argument("--dataset",default="beir/nfcorpus/test");p.add_argument("--pairs",type=int,default=20);p.add_argument("--centroids",type=int,default=256);p.add_argument("--limit-docs",type=int);p.add_argument("--report-dir",type=Path,default=Path("benchmark/reports"));p.add_argument("--cache-dir",type=Path,default=Path("benchmark/cache"));p.add_argument("--refresh-cache",action="store_true");a=p.parse_args()
    dataset=ir_datasets.load(a.dataset);loaded_docs=list(dataset.docs_iter())[:a.limit_docs];docs={d.doc_id:d for d in loaded_docs};queries={q.query_id:q for q in dataset.queries_iter()};pairs=[];seen_queries=set()
    for rel in dataset.qrels_iter():
        if rel.relevance>0 and rel.query_id in queries and rel.doc_id in docs and rel.query_id not in seen_queries:pairs.append((rel.query_id,rel.doc_id));seen_queries.add(rel.query_id)
        if len(pairs)>=a.pairs:break
    model=None
    def encode(values,is_query):
        nonlocal model
        if model is None:model=load_colbert()
        return colbert_encode(model,values,is_query,32)
    qids=list(dict.fromkeys(q for q,_ in pairs));dids=list(dict.fromkeys(d for _,d in pairs));qtexts=[queries[q].text for q in qids];dtexts=[(getattr(docs[d],"title","")+" "+docs[d].text).strip() for d in dids];qvalues,qcache=cached_ragged(a.cache_dir,MODEL_ID,"query",qids,qtexts,lambda:encode(qtexts,True),a.refresh_cache,cache_config("query"));dvalues,dcache=cached_ragged(a.cache_dir,MODEL_ID,"document",dids,dtexts,lambda:encode(dtexts,False),a.refresh_cache,cache_config("document"));qvec=dict(zip(qids,qvalues));dvec=dict(zip(dids,dvalues))
    root=Path(__file__).resolve().parents[1];server=subprocess.Popen(["cargo","run","--release","--bin","multivector","--","--dimension","128","--centroids",str(a.centroids),"--probes","8","--path",str(a.index),"--listen","127.0.0.1:18080"],cwd=root,stdout=subprocess.DEVNULL);base="http://127.0.0.1:18080"
    try:
        for _ in range(60):
            try:http(base,"/healthz");break
            except Exception:time.sleep(1)
        results=[]
        for qid,did in pairs:
            q,d=qvec[qid],dvec[did];reference=rank.rerank(documents_ids=[[did]],queries_embeddings=[q],documents_embeddings=[[d]])[0][0]["score"]
            rust=http(base,"/v1/debug/score",{"query":np.asarray(q).tolist(),"document":np.asarray(d).tolist()})["score"];compressed=http(base,"/v1/debug/score",{"query":np.asarray(q).tolist(),"id":did})["score"];numpy_score=numpy_maxsim(q,d)
            results.append({"query_id":qid,"document_id":did,"reference":reference,"numpy":numpy_score,"rust_uncompressed":rust,"rust_compressed":compressed,"rust_reference_abs_error":abs(rust-reference),"compressed_reference_abs_error":abs(compressed-reference)})
    finally:server.terminate();server.wait(timeout=30)
    report={"dataset":a.dataset,"pairs":len(results),"embedding_cache":{"colbert_queries":qcache,"colbert_documents":dcache},"max_rust_reference_abs_error":max(x["rust_reference_abs_error"] for x in results),"mean_compressed_reference_abs_error":float(np.mean([x["compressed_reference_abs_error"] for x in results])),"max_compressed_reference_abs_error":max(x["compressed_reference_abs_error"] for x in results),"results":results};path=write_report(a.report_dir,"score-validation",report);print(path);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
