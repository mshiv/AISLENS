These are the build and config files required to install MALI and its dependencies.
Software is installed using the spack package manager, and in particular, the E3SM fork on GitHub which includes requirements for albany.

`build.sh` sets the required environment variables, downloads and installs spack, and installs software listed in `env.yaml`.
