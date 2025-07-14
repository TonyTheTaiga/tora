import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
// import adapter from "@sveltejs/adapter-node";
import adapter from "@sveltejs/adapter-vercel";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),

  kit: {
    adapter: adapter({
      // runtime: "nodejs22.x",
      runtime: "edge",
    }),
  },
  csrf: {
    checkOrigin: process.env.NODE_ENV === "production" ? true : false,
  },
};

export default config;
