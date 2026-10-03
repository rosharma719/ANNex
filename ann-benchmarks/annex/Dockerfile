FROM ann-benchmarks-base

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# maturin builds the PyO3 extension
RUN pip install maturin

ARG ANNEX_REPO=https://github.com/rosharma719/ANNex.git
ARG ANNEX_REF=master

RUN git clone --depth 1 --branch "$ANNEX_REF" "$ANNEX_REPO" /annex

WORKDIR /annex/python/annex-py
RUN maturin build --release --features extension-module -o /tmp/annex-wheels
RUN pip install /tmp/annex-wheels/*.whl
