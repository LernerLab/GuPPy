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
through that same port. Port 5006 is the usual default for Bokeh and Panel apps
like GuPPy and has no other significance. Any free port works, as long as you use
the same number in every command below and pass it to GuPPy with `--port`.

## With Docker

On a machine with [Docker](https://docs.docker.com/get-docker/), publish port
5006 and mount the folder that holds your data:

```bash
docker run --rm -p 127.0.0.1:5006:5006 -v /path/to/data:/data \
    ghcr.io/lernerlab/guppy:latest --input-root /data --output-root /data
```

Replace `/path/to/data` with the folder on your computer that holds your data.
Inside the container that folder is `/data`, and GuPPy starts with both its
input and output roots there, so the folder browser shows your own folders. If
your sessions or outputs live in a subfolder, change either root on the GuPPy
homepage.

Then open `http://localhost:5006`. The `127.0.0.1` in `-p`
keeps GuPPy reachable from your own machine only. Keep 5006 on both sides of
`-p`, because GuPPy only accepts connections addressed to the port it serves on.

The image is built for `linux/amd64`. Docker Desktop on an Apple silicon Mac
runs it through emulation, which works but is slower than installing GuPPy
directly.

## On a shared compute cluster with Apptainer

Shared clusters rarely allow Docker, since it needs administrator rights. Most
provide [Apptainer](https://apptainer.org) (formerly Singularity) instead, which
runs the same image without them. Check with `apptainer --version`, or ask your
cluster's support team how to load it.

1. Download the image once into a file you keep:

   ```bash
   apptainer pull guppy.sif docker://ghcr.io/lernerlab/guppy:latest
   ```

2. Start GuPPy on the cluster node where you will analyze the data, making the
   data folder visible inside the container as `/data` with `--bind`:

   ```bash
   apptainer run --bind /path/to/data:/data guppy.sif \
       --address 127.0.0.1 --port 5006 \
       --input-root /data --output-root /data
   ```

   Replace `/path/to/data` with the folder that holds your data. As with Docker,
   GuPPy starts with both roots at `/data`, and you can change either one on the
   GuPPy homepage. `--address 127.0.0.1` keeps GuPPy reachable
   only from the node itself, so other users of the cluster cannot open your
   session. If another user already has port 5006 in use on that node, choose
   another number and use it in the next step too.

3. On your own computer, forward the port through SSH, naming the node GuPPy
   runs on and the login host you normally connect to:

   ```bash
   ssh -L 5006:localhost:5006 -J <user>@<login-host> <user>@<node>
   ```

   Replace `<user>` with your cluster username, `<login-host>` with the address
   you normally `ssh` to, and `<node>` with the name of the node GuPPy runs on,
   which `hostname` prints on that node. If GuPPy runs on the login host itself,
   `ssh -L 5006:localhost:5006 <user>@<login-host>` is enough. Use the same port number on both sides of
   `-L`, because GuPPy only accepts connections addressed to the port it
   serves on.

4. Open `http://localhost:5006` in your browser.
