#!/bin/bash
unset STREAMLIT_SERVER_PORT
unset STREAMLIT_SERVER_ADDRESS
exec streamlit run src/app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true
