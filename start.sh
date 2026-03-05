#!/bin/bash
# Unset Railway's auto-injected variable at runtime before Streamlit reads it
unset STREAMLIT_SERVER_PORT

exec streamlit run src/app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true