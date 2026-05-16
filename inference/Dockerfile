FROM ghcr.io/ggml-org/llama.cpp:server-cuda

COPY scripts/start_server.sh /scripts/start_server.sh
RUN chmod +x /scripts/start_server.sh

EXPOSE 9990
ENTRYPOINT ["/scripts/start_server.sh"]
