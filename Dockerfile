FROM python:3.12-slim

# Chromium and its driver let the visualization dashboard save plots as PNG and SVG.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends chromium chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Without a terminal attached, Python buffers its output, which would hold back the server's startup message.
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE MANIFEST.in /opt/guppy/
COPY src /opt/guppy/src
RUN pip install --no-cache-dir /opt/guppy

# Arguments given to the container are appended, so a later --port overrides this one.
EXPOSE 5006
ENTRYPOINT ["guppy", "--no-browser", "--port", "5006"]
