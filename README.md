# Particles

![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12762669.svg)
![GitHub Release](https://img.shields.io/github/v/release/feelpp/particles)
![CI](https://github.com/feelpp/particles/workflows/CI/badge.svg)

A template repository to kickstart your finite-element applications with Feel++.

## Table of Contents

* [Features](#features)
* [Quick Start](#quick-start)
* [Building](#building)
* [Continuous Integration](#ci)
* [Versioning & Release](#versioning)
* [Contributing](#contributing)
* [License](#license)

## Features

* ***C++*** example apps using Feel++ and its toolboxes (located in `src/`).
* ***Python*** Jupyter notebooks under `docs/notebooks/` for interactive demos.
* ***Documentation*** authored in AsciiDoc and published with Antora (`docs/`).
* ***Docker*** setup for reproducible development and deployment.
* ***CI*** via GitHub Actions: C++ tests, Python wheel builds, docs site.

## Quick Start

### Prerequisites

* CMake ≥ 3.21  
* A C++ compiler (GCC or Clang) with MPI support  
* Python 3.8+ and `pip`  
* Docker (optional, for container builds)

### Clone & Rename

If you used this as a template, rename the project metadata:
```bash
./rename.sh your-project-name
```

**📌 NOTE**\
After renaming, verify URLs in `docs/site.yml` and `docs/package.json`.

## Building

### CMake Presets

Create ***CMakePresets.json*** in your project root or update the one provided:

```bash
cmake --preset default
cmake --build --preset default
cmake --build --preset default --target install
```

`build/default` will contain the build artifacts and the build directory.

## Continuous Integration

Our GitHub Actions workflow (`.github/workflows/ci.yml`) includes:

* build_wheel: Python wheel compilation and artifact upload.
* docs: Builds the Antora site, deploys to GitHub Pages on master.
* build_code: CMake build, tests with ctest --preset default, packaging.
* deliver: Docker image build & push to GHCR.
* release: On tags vX.Y.Z, publishes binaries, wheels, datasets, and creates a GitHub release.

## Versioning & Release

Project version is centrally defined in:

* CMakeLists.txt
* docs/antora.yml
* docs/package.json

### Release Process

**	Commit with:**

+
 $ git commit -am "Release vX.Y.Z"
+
. Tag and push:
+
 $ git tag vX.Y.Z && git push --tags
+
. GitHub Actions will build and publish artifacts automatically.

## Contributing

We welcome contributions! Please:

* Fork the repository and create a feature branch.
* Adhere to existing coding conventions; add C++ tests where appropriate.
* Update documentation (docs/) for any new features.
* Submit a pull request with a clear description of your changes.

## License

This project is licensed under the BSD 3-Clause License.
See LICENSE for full details.
