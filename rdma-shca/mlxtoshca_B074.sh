apt-get update -y \
&& apt remove -y  rdmacm-utils || true \
&& apt remove -y ibacm || true \
&& apt remove -y  perftest || true \
&& apt remove -y ibverbs-utils || true \
&& apt remove -y ucx || true \
&& apt remove -y libibverbs-dev || true \
&& apt remove -y libibmad-dev || true \
&& apt remove -y libibumad-dev || true \
&& apt remove -y librdmacm1 || true \
&& apt remove -y infiniband-diags || true \
&& apt remove -y opensm || true \
&& apt remove -y rdma-core || true \
&& apt remove -y libibmad5 || true \
&& apt remove -y libibumad3 || true \
&& apt remove -y ibverbs-providers  || true \
&& apt remove -y libibverbs1 || true \
&& apt install -y libmosquitto1 || true \
        && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && dpkg -i  shca-tools_2.500.4.B074-Ubuntu22.04_amd64.deb
