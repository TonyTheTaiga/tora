import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import adapter from "@sveltejs/adapter-node";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),

  kit: {
    adapter: adapter(),
  },
  csrf: {
    checkOrigin: process.env.NODE_ENV === "production" ? true : false,
  },
};

export default config;
