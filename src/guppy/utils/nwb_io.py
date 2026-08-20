"""Read and write NWB files against the schema they were written with.

hdmf keeps one type map per process. Importing an NWB extension registers the installed version of
that extension there, and from then on a file whose cached namespace holds a *different* version is
built against the installed spec rather than its own -- a ``ConstructError`` for any type whose
layout changed between the two. GuPPy imports ndx-guppy, and with it ndx-fiber-photometry, partway
through an NWB export, so every read after the first export in a process is exposed: the second
session of an export batch, or a return to Step 1 in the long-lived GUI.

Reads therefore go through a type map seeded from a snapshot of pynwb's map taken at import time,
into which only the file's own cached namespaces are loaded. Nothing registered later in the process
can shadow them.

The snapshot has to be taken before any extension registers itself, so every GuPPy module that reads
NWB imports this one at module scope, while ndx-guppy is reached only from inside a conversion call.

A file read this way is written back through the same map, keeping the output on the extension
versions its source was written with. It has to be: hdmf builds an export with the manager that read
the container and accepts no other, so those versions are the only ones the output can hold whatever
is installed here. The extensions GuPPy adds on the way out are registered into that map first.
"""

import os
from copy import deepcopy
from glob import glob
from importlib import import_module
from pathlib import Path

import h5py
import yaml
from hdmf.build import BuildManager, TypeMap
from pynwb import NWBHDF5IO, NWBFile, get_type_map

# Taken at import, when the global map still holds core, hdmf-common and hdmf-experimental alone:
# GuPPy pulls in pynwb long before neuroconv imports any ndx-* extension. ``get_type_map`` already
# returns a deep copy, so later registrations do not reach this one.
_CORE_TYPE_MAP = get_type_map()


def open_nwbfile_io(*, path: str | None = None, file: h5py.File | None = None) -> NWBHDF5IO:
    """Open an NWB file for reading, built against the namespaces cached inside it.

    Parameters
    ----------
    path : str, optional
        Path to the ``.nwb`` file. Give this or ``file``.
    file : h5py.File, optional
        An already-open HDF5 file, as the DANDI streaming reader holds. Give this or ``path``.

    Returns
    -------
    pynwb.NWBHDF5IO
        An open IO in read mode. The caller owns it and must close it.
    """
    type_map = deepcopy(_CORE_TYPE_MAP)
    NWBHDF5IO.load_namespaces(type_map, path=path, file=file)
    return NWBHDF5IO(path=path, file=file, manager=BuildManager(type_map))


def _register_namespace(*, type_map: TypeMap, namespace: str) -> None:
    """Add an installed extension's namespace and container classes to ``type_map``.

    The namespace is loaded from the installed package's own spec, so the extensions it depends on
    are not pulled in at whatever version this process has: one already in the catalog keeps the
    version it was loaded with. Only the types the extension itself defines are registered -- a
    namespace also lists everything it inherits from those dependencies, whose classes belong to
    the versions already in the map.
    """
    module = import_module(namespace.replace("-", "_"))
    specification_directory = os.path.join(os.path.dirname(module.__file__), "spec")
    (namespace_path,) = glob(os.path.join(specification_directory, "*namespace.yaml"))
    type_map.load_namespaces(namespace_path)

    installed_type_map = get_type_map()
    for source_file in type_map.namespace_catalog.get_namespace(namespace).get_source_files():
        schema = yaml.safe_load(Path(specification_directory, source_file).read_text())
        for entries in schema.values():
            for entry in entries:
                data_type = entry.get("neurodata_type_def")
                if data_type is None:
                    continue
                container_class = installed_type_map.get_dt_container_cls(data_type, namespace)
                type_map.register_container_type(namespace, data_type, container_class)


def write_nwbfile_from_source(
    *,
    nwbfile: NWBFile,
    source_io: NWBHDF5IO,
    nwbfile_path: str,
    added_namespaces: tuple[str, ...] = (),
) -> None:
    """Write a file read by ``source_io`` to ``nwbfile_path``, on the schema it was read with.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The file ``source_io`` read, with whatever has since been added to it.
    source_io : pynwb.NWBHDF5IO
        The still-open IO that read ``nwbfile``. hdmf builds the export with this IO's manager and
        rejects any other, which is what keeps the output on the source's extension versions.
    nwbfile_path : str
        Path to write. Writing to a new path leaves the source untouched.
    added_namespaces : tuple of str, optional
        Namespaces of extensions whose containers were added to ``nwbfile`` after it was read, and
        which the source therefore does not know about -- ``("ndx-guppy",)`` for a GuPPy export.
    """
    from neuroconv.tools.nwb_helpers import (
        configure_backend,
        get_default_backend_configuration,
    )

    type_map = source_io.manager.type_map
    for namespace in added_namespaces:
        _register_namespace(type_map=type_map, namespace=namespace)

    configure_backend(nwbfile=nwbfile, backend_configuration=get_default_backend_configuration(nwbfile, "hdf5"))
    nwbfile.set_modified()
    with NWBHDF5IO(nwbfile_path, mode="w", manager=BuildManager(type_map)) as io:
        io.export(nwbfile=nwbfile, src_io=source_io, write_args=dict(link_data=False))
