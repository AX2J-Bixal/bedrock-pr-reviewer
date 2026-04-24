// src/bot.ts
import {
  BedrockRuntimeClient,
  ConversationRole,
  ConverseCommand,
  ConverseCommandInput,
  ConverseCommandOutput,
  ToolConfiguration
} from '@aws-sdk/client-bedrock-runtime'
import {info, warning} from '@actions/core'
import pRetry from 'p-retry'
import {BedrockOptions, Options} from './options'

// define type to save parentMessageId and conversationId
export interface Ids {
  parentMessageId?: string
  conversationId?: string
}

export interface JsonSchema {
  name: string
  description: string
  parameters: Record<string, any>
}

export class Bot {
  private readonly client: BedrockRuntimeClient

  private readonly options: Options
  private readonly bedrockOptions: BedrockOptions

  // Opus 4.7+ and other thinking models reject temperature=0.
  // Learn from first failure or pre-set for known models.
  private temperatureRejected = false

  constructor(options: Options, bedrockOptions: BedrockOptions) {
    this.options = options
    this.bedrockOptions = bedrockOptions
    this.client = new BedrockRuntimeClient({})

    // Opus 4.7+ with adaptive thinking doesn't support temperature.
    // The API hangs instead of returning an error, so we preemptively disable it.
    if (bedrockOptions.model.includes('opus-4-7')) {
      this.temperatureRejected = true
    }
  }

  chat = async (
    message: string,
    jsonSchema?: JsonSchema
  ): Promise<[string, Ids]> => {
    let res: [string, Ids] = ['', {}]
    try {
      res = await this.chat_(message, jsonSchema)
      return res
    } catch (e: unknown) {
      warning(`Failed to chat: ${e}`)
      return res
    }
  }

  private readonly chat_ = async (
    message: string,
    jsonSchema?: JsonSchema
  ): Promise<[string, Ids]> => {
    const start = Date.now()
    if (!message) {
      return ['', {}]
    }

    let response: ConverseCommandOutput | undefined

    message = `IMPORTANT: Entire response must be in the language with ISO code: ${this.options.language}\n\n${message}`

    if (this.options.debug) {
      info(`sending prompt: ${message}\n------------`)
      if (jsonSchema) {
        info(`Using JSON schema: ${JSON.stringify(jsonSchema)}`)
      }
    }

    // Rebuilt on every retry so temperatureRejected flag changes take effect
    const buildParams = (): ConverseCommandInput => {
      const params: ConverseCommandInput = {
        modelId: this.bedrockOptions.model,
        messages: [
          {
            role: 'user' as ConversationRole,
            content: [
              {
                text: message
              }
            ]
          }
        ],
        inferenceConfig: {
          maxTokens: 4096,
          ...(this.temperatureRejected ? {} : {temperature: 0})
        }
      }

      // Opus 4.7+ requires adaptive thinking configuration.
      // Without this, the Converse API call hangs indefinitely.
      // See: https://aws.amazon.com/blogs/aws/introducing-anthropics-claude-opus-4-7-model-in-amazon-bedrock/
      if (this.bedrockOptions.model.includes('opus-4-7')) {
        params.additionalModelRequestFields = {
          thinking: {
            type: 'adaptive',
            budget_tokens: 10000
          }
        }
        // Thinking models need higher output limit
        params.inferenceConfig!.maxTokens = 16384
      }

      // Add tool configuration if jsonSchema is provided
      if (jsonSchema) {
        const toolConfig: ToolConfiguration = {
          tools: [
            {
              toolSpec: {
                name: jsonSchema.name,
                description: jsonSchema.description,
                inputSchema: {
                  json: jsonSchema.parameters
                }
              }
            }
          ]
        }
        params.toolConfig = toolConfig
      }

      return params
    }

    const attempt = async (): Promise<ConverseCommandOutput> => {
      const params = buildParams()

      if (this.options.debug) {
        info(`Bedrock request params: ${JSON.stringify(params, null, 2)}`)
      }

      try {
        return await this.client.send(new ConverseCommand(params))
      } catch (e: any) {
        // Bedrock returns ValidationException when temperature is not supported.
        // Flip the flag and rethrow so pRetry rebuilds params without temperature.
        if (
          e?.name === 'ValidationException' &&
          typeof e?.message === 'string' &&
          (e.message.includes('temperature') ||
            e.message.includes('inferenceConfig')) &&
          !this.temperatureRejected
        ) {
          warning(
            `${this.bedrockOptions.model} rejected temperature — retrying without it`
          )
          this.temperatureRejected = true
        }
        throw e
      }
    }

    try {
      response = await pRetry(attempt, {
        retries: this.options.bedrockRetries
      })
    } catch (e: any) {
      warning(
        `bedrock send failed: name=${e?.name} message=${e?.message} fault=${e?.$fault} requestId=${e?.$metadata?.requestId}`
      )
    }

    const end = Date.now()
    info(
      `bedrock sendMessage (including retries) response time: ${end - start} ms`
    )

    let responseText = ''
    if (response?.output?.message != null) {
      const content = response.output.message.content || []
      for (const item of content) {
        if (item.text) {
          responseText += item.text
        } else if (item.toolUse) {
          try {
            responseText = JSON.stringify(item.toolUse.input)
          } catch (e) {
            warning(`Failed to parse tool use input as JSON: ${e}`)
            responseText = ''
          }
        }
      }
    } else {
      warning('bedrock response is null')
    }

    if (this.options.debug) {
      info(`bedrock responses: ${responseText}\n-----------`)
    }

    const newIds: Ids = {
      parentMessageId: response?.$metadata.requestId,
      conversationId: response?.$metadata.cfId
    }
    return [responseText, newIds]
  }
}
