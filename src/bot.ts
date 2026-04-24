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

  // Some newer Bedrock models (observed on Opus 4.7) reject temperature=0
  // with a ValidationException. We learn from the first failure and reuse
  // temperature=1 for the life of this Bot instance.
  private temperatureRejected = false

  constructor(options: Options, bedrockOptions: BedrockOptions) {
    this.options = options
    this.bedrockOptions = bedrockOptions
    this.client = new BedrockRuntimeClient({})
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
    // record timing
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

    // Rebuilt on every retry attempt so a flipped `temperatureRejected`
    // flag is picked up on the next pRetry iteration.
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
          temperature: this.temperatureRejected ? 1 : 0
        }
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
      try {
        return await this.client.send(new ConverseCommand(buildParams()))
      } catch (e: any) {
        // Bedrock returns ValidationException with the literal string
        // "temperature is deprecated for this model." on Opus 4.7+ when
        // temperature=0 is sent. Flip the flag and rethrow so pRetry
        // rebuilds params with temperature=1 on the next attempt.
        if (
          e?.name === 'ValidationException' &&
          typeof e?.message === 'string' &&
          e.message.includes('temperature is deprecated') &&
          !this.temperatureRejected
        ) {
          warning(
            `${this.bedrockOptions.model} rejected temperature=0 — retrying with temperature=1`
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
      // Was previously `info(\`...: ${e}\`)`, which stringified the error
      // and dropped name/fault/message/requestId. That's how the Opus 4.7
      // failure looked like an indefinite hang — a clean ValidationException
      // was being silently swallowed. Log explicit fields instead.
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
      // Check if the response contains a tool use (JSON output)
      const content = response.output.message.content || []
      for (const item of content) {
        if (item.text) {
          responseText += item.text
        } else if (item.toolUse) {
          // For JSON schema tool use, the input will contain the generated JSON
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
      info(`bedrock responses: ${responseText}\n—————`)
    }
    const newIds: Ids = {
      parentMessageId: response?.$metadata.requestId,
      conversationId: response?.$metadata.cfId
    }
    return [responseText, newIds]
  }
}