{pkgs}: {
  channel = "stable-24.05";
  packages = [
    pkgs.nodejs_20
    pkgs.pnpm
    pkgs.python313Full
    pkgs.uv
  ];
  idx.extensions = [
    "svelte.svelte-vscode"
    "vue.volar"
  ];
  idx.previews = {
    previews = {
      web = {
        command = ["sh" "-c" "cd web && npm run dev -- --port $PORT --host 0.0.0.0"];
        manager = "web";
      };
    };
  };
}
