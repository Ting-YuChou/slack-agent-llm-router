import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

import { buildGatewayProviderRegistrations } from "../dist/src/agent-model.js";

export default function modelGatewayExtension(pi: ExtensionAPI) {
  const gatewayUrl = process.env.PI_MODEL_GATEWAY_URL;
  if (!gatewayUrl || !/^http:\/\/model-gateway:\d+$/.test(gatewayUrl)) {
    throw new Error("PI_MODEL_GATEWAY_URL must target the internal model gateway root");
  }
  for (const registration of buildGatewayProviderRegistrations(gatewayUrl)) {
    const { models } = registration;
    const modelConfigs: ProviderModelConfig[] = models.map((model) => ({
      id: model.id,
      name: model.name,
      api: model.api,
      reasoning: true,
      thinkingLevelMap: model.thinkingLevelMap,
      input: model.input,
      cost: model.cost,
      contextWindow: model.contextWindow,
      maxTokens: model.maxTokens,
      compat: model.compat,
    }));
    pi.registerProvider(registration.provider, {
      baseUrl: registration.baseUrl,
      apiKey: registration.apiKey,
      api: registration.api,
      models: modelConfigs,
    });
  }
}
