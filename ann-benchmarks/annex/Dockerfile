FROM ann-benchmarks-base

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# maturin builds the PyO3 extension
RUN pip install maturin

ARG ANNEX_REPO=https://github.com/rosharma719/ANNex.git
ARG ANNEX_REF=df9da0ff6bc878c0f7c48ed8c33aed57ba8e2040

RUN git init /annex \
    && cd /annex \
    && git remote add origin "$ANNEX_REPO" \
    && git fetch --depth 1 origin "$ANNEX_REF" \
    && git checkout --detach FETCH_HEAD

WORKDIR /annex/python/annex-py
RUN maturin build --release --features extension-module -o /tmp/annex-wheels
RUN pip install /tmp/annex-wheels/*.whl
