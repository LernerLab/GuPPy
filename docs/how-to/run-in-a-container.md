# Run GuPPy in a container

GuPPy is published as a container image, `ghcr.io/lernerlab/guppy`, with GuPPy
and everything it needs already installed. Use it when you cannot install
software on the machine that holds your data, such as a shared compute cluster,
or when you want a fixed GuPPy version without managing a Python environment.
To install GuPPy directly instead, see [Installation](../installation.md).

The image is tagged with each GuPPy release (for example `v2.0.0-beta5`), and
`latest` points at the most recent release. Name a release tag when you want
the same GuPPy version every time you start it.

GuPPy runs as a web server inside the container. You open it in your own
browser at `http://localhost:5006`, and every page GuPPy opens reaches you
through that same port.

## On a shared compute cluster with Apptainer

Most clusters provide [Apptainer](https://apptainer.org) (formerly Singularity),
which runs the image without administrator rights. Check with `apptainer
--version`, or ask your cluster's support team how to load it.

1. Download the image once into a file you keep:

   ```bash
   apptainer pull guppy.sif docker://ghcr.io/lernerlab/guppy:latest
   ```

2. Start GuPPy on the cluster node where you will analyze the data, making the
   data folders visible inside the container with `--bind`:

   ```bash
   apptainer run --bind /path/to/data guppy.sif \
       --address 127.0.0.1 --port 5006 \
       --input-root /path/to/data/sessions --output-root /path/to/data/outputs
   ```

   Apptainer already makes your home directory visible, so `--bind` is only
   needed for folders outside it. `--address 127.0.0.1` keeps GuPPy reachable
   only from the node itself, so other users of the cluster cannot open your
   session. If another user already has port 5006 in use on that node, choose
   another number and use it in the next step too.

3. On your own computer, forward the port through SSH, naming the node GuPPy
   runs on and the login host you normally connect to:

   ```bash
   ssh -L 5006:localhost:5006 -J <user>@<login host> <user>@<node>
   ```

   If GuPPy runs on the login host itself, `ssh -L 5006:localhost:5006
   <user>@<login host>` is enough. Use the same port number on both sides of
   `-L`, because GuPPy only accepts connections addressed to the port it
   serves on.

4. Open `http://localhost:5006` in your browser.

## With Docker

On a machine with [Docker](https://docs.docker.com/get-docker/), publish port
5006 and mount the folder that holds your data:

```bash
docker run --rm -p 127.0.0.1:5006:5006 -v /path/to/data:/data \
    ghcr.io/lernerlab/guppy:latest --input-root /data/sessions --output-root /data/outputs
```

Then open `http://localhost:5006`. Inside the container your data folder is
`/data`, so every path you give GuPPy starts there. The `127.0.0.1` in `-p`
keeps GuPPy reachable from your own machine only. Keep 5006 on both sides of
`-p`, because GuPPy only accepts connections addressed to the port it serves on.

The image is built for `linux/amd64`. Docker Desktop on an Apple silicon Mac
runs it through emulation, which works but is slower than installing GuPPy
directly.

## Opening pages

Each step that shows a page, such as **Step 1: Label Stores** or a result view
after **Step 3: Preprocess**, opens it in a new browser tab. If your browser
blocks the tab, GuPPy shows a notification on the homepage with a link to the
page. This happens most often after a long step finishes. Allow pop-ups for
`localhost` to have tabs open on their own.

## Streaming from DANDI

To [analyze data streamed from the DANDI Archive](analyze-dandi-data.md), pass
your API key into the container, since a key stored with `dandi login` on your
computer is not visible inside it. With Apptainer, export `DANDI_API_KEY` before
`apptainer run` and it is passed through. With Docker, add
`-e DANDI_API_KEY` to `docker run`.
