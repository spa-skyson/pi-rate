{
  description = "pi-go, an extensible AI coding agent";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          version = "unstable-${self.shortRev or "dirty"}";
          package = pkgs.buildGoModule {
            pname = "pi-go";
            inherit version;
            src = ./.;
            # vendorHash is stale after the 2026-10 go.mod bump
            # (x/* upgrades); the first `nix build` must update it —
            # Nix will print the correct hash. Do not guess.
            vendorHash = "sha256-43goJ21uDYZbTOwELp7V3IqxeGhWi5wmIgP6qLy4G4Q=";

            # The repository currently requires Go 1.27, which is not yet
            # available in the pinned nixpkgs revision. This is only a module
            # directive compatibility patch; the source uses no newer APIs.
            #
            # This was once accompanied by a second substitution promoting
            # google.golang.org/grpc out of the indirect block. main now
            # requires it directly, so the pattern no longer matches — and
            # because substituteInPlace uses --replace-fail, a stale pattern
            # does not degrade to a no-op, it aborts the build.
            postPatch = ''
              substituteInPlace go.mod \
                --replace-fail 'go 1.27.2' 'go 1.26.5'
            '';

            subPackages = [ "cmd/pi" ];
            buildFlags = [ "-mod=mod" ];
            ldflags = [
              "-s"
              "-w"
              "-X github.com/spa-skyson/pi-rate/internal/cli.Version=${version}"
              "-X github.com/spa-skyson/pi-rate/internal/cli.BuildTag=${self.shortRev or "source"}"
            ];

            meta = {
              description = "An extensible AI coding agent";
              homepage = "https://github.com/spa-skyson/pi-rate";
              license = pkgs.lib.licenses.mit;
              mainProgram = "pi";
            };
          };
        in {
          default = package;
        });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/pi";
        };
      });

      devShells = forAllSystems (system:
        let pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = [ pkgs.go_1_26 pkgs.git pkgs.gnumake ];
          };
        });

      nixosModules.default = { config, lib, pkgs, ... }:
        with lib;
        let
          cfg = config.programs.pi-go;
        in {
          options.programs.pi-go = {
            enable = mkEnableOption "pi-go";

            package = mkOption {
              type = types.package;
              default = self.packages.${pkgs.system}.default;
              defaultText = literalExpression "self.packages.\${pkgs.system}.default";
              description = "The pi-go package to install.";
            };
          };

          config = mkIf cfg.enable {
            environment.systemPackages = [ cfg.package ];
          };
        };
    };
}
