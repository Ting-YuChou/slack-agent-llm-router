/** Agent image the opt-in Docker suites run; CI points it at the image it just built. */
export const AGENT_TEST_IMAGE = process.env.PI_AGENT_TEST_AGENT_IMAGE ?? "slack-pi-agent:1.0.1";
