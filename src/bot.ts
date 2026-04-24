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

  // Some models (Opus 4.7+) reject temperature parameter
  private temperatureRejected = false

  constructor(options: Options, bedrockOptions: BedrockOptions) {
    this.options = options
    this.bedrockOptions = bedrockOptions
    this.client = new BedrockRuntimeClient({})

    // Opus 4.7 doesn't support temperature — causes API to hang
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

    const buildParams = (): ConverseCommandInput => {
      const params: ConverseCommandInput = {
        modelId: this.bedrockOptions.model,
        messages: [
          {
            role: 'user' as ConversationRole,
            content: [{ text: message }]
          }
        ],
        inferenceConfig: {
          maxTokens: 4096,
          ...(this.temperatureRejected ? {} : {temperature: 0})
        }
      }

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
