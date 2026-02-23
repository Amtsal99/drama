import { NextResponse } from 'next/server'
import { CONFIG } from '@/lib/config'
import { generateWithModel } from '@/lib/models'
import type { Report } from '@/types'
import { reportContentRatelimit } from '@/lib/redis'
//for cost estimation
import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";
import { addCost, getTotalCost } from '@/app/api/costTracker';

const PRICE_PER_TOKEN_INPUT_GEMINI_2_5_FLASH = 0.0000003
const PRICE_PER_TOKEN_OUTPUT_GEMINI_2_5_FLASH = 0.000025

export async function POST(request: Request) {
  try {
    const { reports, platformModel } = await request.json()
    const [platform, model] = platformModel.split('__')

    if (CONFIG.rateLimits.enabled && platform !== 'ollama') {
      const { success } = await reportContentRatelimit.limit('report')
      if (!success) {
        return NextResponse.json(
          { error: 'Too many requests' },
          { status: 429 }
        )
      }
    }

    console.log('Consolidating reports:', {
      numReports: reports.length,
      reportTitles: reports.map((r: Report) => r.title),
      platform,
      model,
    })

    if (!reports?.length) {
      return NextResponse.json(
        { error: 'Reports are required' },
        { status: 400 }
      )
    }

    // Collect all unique sources from all reports
    const allSources: { id: string; url: string; name: string }[] = []
    const sourceMap = new Map<string, number>() // Maps source id to index in allSources

    reports.forEach((report: Report) => {
      if (report.sources && report.sources.length > 0) {
        report.sources.forEach((source) => {
          if (!sourceMap.has(source.id)) {
            sourceMap.set(source.id, allSources.length)
            allSources.push(source)
          }
        })
      }
    })

    // Create source index for citations
    const sourceIndex = allSources
      .map(
        (source, index) =>
          `[${index + 1}] Source: ${source.name} - ${source.url}`
      )
      .join('\n')

    const prompt = `Create a comprehensive consolidated report that synthesizes the following research reports:

${reports
  .map(
    (report: Report, index: number) => `
Report ${index + 1} Title: ${report.title}
Report ${index + 1} Summary: ${report.summary}
Key Findings:
${report.sections
  ?.map((section) => `- ${section.title}: ${section.content}`)
  .join('\n')}
`
  )
  .join('\n\n')}

Sources for citation:
${sourceIndex}

Analyze and synthesize these reports to create a comprehensive consolidated report that:
1. Identifies common themes and patterns across the reports
2. Highlights key insights and findings
3. Shows how different reports complement or contrast each other
4. Draws overarching conclusions
5. Suggests potential areas for further research
6. Uses citations only when necessary to reference specific claims, statistics, or quotes from sources

Format the response as a structured report with:
- A clear title that encompasses the overall research topic
- An executive summary of the consolidated findings
- Detailed sections that analyze different aspects
- A conclusion that ties everything together
- Judicious use of citations in superscript format [¹], [²], etc. ONLY when necessary

Return the response in the following JSON format:
{
  "title": "Overall Research Topic Title",
  "summary": "Executive summary of findings",
  "sections": [
    {
      "title": "Section Title",
      "content": "Section content with selective citations"
    }
  ],
  "usedSources": [1, 2] // Array of source numbers that were actually cited in the report
}

CITATION GUIDELINES:
1. Only use citations when truly necessary - specifically for:
   - Direct quotes from sources
   - Specific statistics, figures, or data points
   - Non-obvious facts or claims that need verification
   - Controversial statements
   
2. DO NOT use citations for:
   - General knowledge
   - Your own analysis or synthesis of information
   - Widely accepted facts
   - Every sentence or paragraph

3. When needed, use superscript citation numbers in square brackets [¹], [²], etc. at the end of the relevant sentence
   
4. The citation numbers correspond directly to the source numbers provided in the list
   
5. Be judicious and selective with citations - a well-written report should flow naturally with citations only where they truly add credibility

6. You DO NOT need to cite every source provided. Only cite the sources that contain information directly relevant to the report. Track which sources you actually cite and include their numbers in the "usedSources" array in the output JSON.

7. It's completely fine if some sources aren't cited at all - this means they weren't needed for the specific analysis requested.`

    console.log('Generated prompt:', prompt)

    try {
      const res = await generateWithModel(prompt, platformModel)
                        
      const input_token = res.usageMetadata?.promptTokenCount ?? 0.0;
      const output_token = res.usageMetadata?.candidatesTokenCount ?? 0.0;
      console.log("INPUT COST: ", input_token * PRICE_PER_TOKEN_INPUT_GEMINI_2_5_FLASH);
      console.log("OUTPUT COST: ", output_token * PRICE_PER_TOKEN_OUTPUT_GEMINI_2_5_FLASH);
      console.log("before cost: ", getTotalCost())
      const cost = input_token * PRICE_PER_TOKEN_INPUT_GEMINI_2_5_FLASH + output_token * PRICE_PER_TOKEN_OUTPUT_GEMINI_2_5_FLASH;
      addCost(cost)
      console.log("after cost: ", getTotalCost())

      const response = res.text ?? ""
            // Try to parse the response as JSON, if it's not already
      let parsedResponse
      try {
        parsedResponse =
          typeof response === 'string' ? JSON.parse(response) : response
        console.log('Parsed response:', parsedResponse)
      } catch (parseError) {
        console.error('Failed to parse response as JSON:', parseError)
        // If it's not JSON, create a basic report structure
        parsedResponse = {
          title: 'Consolidated Research Report',
          summary: response.split('\n\n')[0] || 'Summary not available',
          sections: [
            {
              title: 'Findings',
              content: response,
            },
          ],
        }
      }

      // Add sources to the response
      parsedResponse.sources = allSources

      return NextResponse.json(parsedResponse)
    } catch (error) {
      console.error('Model generation error:', error)
      return NextResponse.json(
        { error: 'Failed to generate consolidated report' },
        { status: 500 }
      )
    }
  } catch (error) {
    console.error('Consolidation error:', error)
    return NextResponse.json(
      { error: 'Failed to consolidate reports' },
      { status: 500 }
    )
  }
}
