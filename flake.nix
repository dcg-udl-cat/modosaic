{
  description = "A flake configuration to run Modosaic's CUDA code in a dev shell with the necessary dependencies.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nix-gl-host = {
      url = "github:numtide/nix-gl-host";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "https://nix-community.cachix.org"
      "https://cache.nixos-cuda.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "nix-community.cachix.org-1:mB9ZQ+4kTq9qUqM96H8P6oz+ZWHR+Hh3wlgYx9oSt1A="
      "cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="
    ];
  };

  outputs =
    {
      self,
      nixpkgs,
      nix-gl-host,
    }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
        config.cudaSupport = true;
      };
    in
    {
      devShells.x86_64-linux.default =
        with pkgs;
        mkShell rec {
          name = "modosaic-cuda-dev-shell";
          packages = [
            cmake
            ninja
            wget
            stdenv.cc.cc
            stdenv.cc.cc.lib
            libGL
            xorg.libX11
            xorg.libXext
            xorg.libXrender
            xorg.libxcb
            xorg.libXi
            xorg.libXrandr
            xorg.libXcursor
            xorg.libXinerama
            xorg.libXfixes
            glib
            fontconfig
            freetype
            dbus
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart
            cudaPackages.cuda_cupti
            cudaPackages.cuda_nvrtc
            cudaPackages.cuda_nvtx
            cudaPackages.cudnn
            cudaPackages.libcublas
            cudaPackages.libcufft
            cudaPackages.libcurand
            cudaPackages.libcusolver
            cudaPackages.libcusparse
            cudaPackages.libnvjitlink
            cudaPackages.nccl
            nix-gl-host.defaultPackage.x86_64-linux
            uv
            python313
            zlib
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${lib.makeLibraryPath packages}:$(nixglhost -p):$LD_LIBRARY_PATH"
             if [ ! -f model_weights/sam_vit_b_01ec64.pth ]; then
              echo "SAM ViT-B weights not found. Downloading..."
              mkdir -p model_weights
              wget -O model_weights/sam_vit_b_01ec64.pth \
                https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

            fi
            uv sync
            . .venv/bin/activate
            echo "Entering CUDA dev shell, python venv is ready and activated."
          '';
        };
    };
}
