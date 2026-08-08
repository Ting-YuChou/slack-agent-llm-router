import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

import { gatewayProviderConfig, listAgentModels } from "../dist/src/agent-model.js";

export default function modelGatewayExtension(pi: ExtensionAPI) {
  const gatewayUrl = process.env.PI_MODEL_GATEWAY_URL;
  if (!gatewayUrl || !/^http:\/\/model-gateway:\d+$/.test(gatewayUrl)) {
    throw new Error("PI_MODEL_GATEWAY_URL must target the internal model gateway root");
  }
  for (const model of listAgentModels()) {
    const gateway = gatewayProviderConfig(gatewayUrl, model.ref);
    const modelConfig: ProviderModelConfig = {
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
    };
    pi.registerProvider(model.provider, {
      baseUrl: gateway.baseUrl,
      apiKey: gateway.apiKey,
      api: gateway.api,
      models: [modelConfig],
    });
  }
}
