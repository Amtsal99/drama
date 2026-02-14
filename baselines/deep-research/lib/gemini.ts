import {
  GoogleGenAI,
  SafetySetting,
  HarmCategory, 
  HarmBlockThreshold
} from '@google/genai'

const safetySettings: SafetySetting[] = [
  {
    category: HarmCategory.HARM_CATEGORY_HARASSMENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
  {
    category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    threshold: HarmBlockThreshold.BLOCK_NONE,
  },
];


const generationJsonConfig = {
  temperature: 1,
  maxOutputTokens: 8192,
  responseMimeType: 'application/json',
};

const generationPlainTextConfig = {
  temperature: 1,
  maxOutputTokens: 8192,
  responseMimeType: 'text/plain',
};

const genAI = new GoogleGenAI({apiKey: process.env.GEMINI_API_KEY || ''})

export const geminiFlashLiteModel = (contents: any) => 
  genAI.models.generateContent({
    model: 'gemini-2.0-flash-lite-preview-02-05',
    contents,
    config: { ...generationJsonConfig, safetySettings }
  });

export const geminiFlashModel = (contents: any) => 
  genAI.models.generateContent({
    model: 'gemini-2.5-flash',
    contents,
    config: { ...generationJsonConfig, safetySettings }
  });

export const geminiFlashThinkingModel = (contents: any) => 
  genAI.models.generateContent({
    model: 'gemini-2.0-flash-thinking-exp-01-21',
    contents,
    config: { ...generationPlainTextConfig, safetySettings }
  });

export const geminiModel = (contents: any) => 
  genAI.models.generateContent({
    model: 'gemini-2.0-pro-exp-02-05',
    contents,
    config: { ...generationJsonConfig, safetySettings }
  });