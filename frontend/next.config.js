/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack(config) {
    config.experiments = { ...config.experiments, asyncWebAssembly: true };
    return config;
  },
  headers: async () => [
    {
      source: "/models/:path*",
      headers: [
        { key: "Cache-Control", value: "public, max-age=31536000, immutable" }
      ]
    }
  ]
};

module.exports = nextConfig;