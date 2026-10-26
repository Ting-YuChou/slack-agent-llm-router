/** Agent image the opt-in Docker suites run; CI points it at the image it just built. */
export const AGENT_TEST_IMAGE = process.env.PI_AGENT_TEST_AGENT_IMAGE ?? "slack-pi-agent:1.0.1";

/** Plain Node image the fake model and MCP gateways run in, pinned to the same digest as the Dockerfiles' Node base. */
export const FAKE_GATEWAY_IMAGE = "node:22.22.0-bookworm-slim@sha256:dd9d21971ec4395903fa6143c2b9267d048ae01ca6d3ea96f16cb30df6187d94";
