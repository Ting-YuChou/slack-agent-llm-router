import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function modelGatewayExtension(pi: ExtensionAPI) {
  const baseUrl = process.env.PI_MODEL_GATEWAY_URL;
  if (!baseUrl || !baseUrl.startsWith("http://model-gateway:")) {
    throw new Error("PI_MODEL_GATEWAY_URL must target the internal model gateway");
  }
  pi.registerProvider("openai", {
    baseUrl,
    apiKey: "$OPENAI_API_KEY",
  });
}
