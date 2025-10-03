import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 30;

interface PropertyRAGMessage {
  role: "user" | "assistant";
  content: string | Array<{ type: string; text: string; [key: string]: any }>;
}

export async function POST(req: NextRequest) {
  try {
    console.log("API route called");
    const { messages }: { messages: PropertyRAGMessage[] } = await req.json();
    console.log("Messages received:", messages);
    
    // Get the latest user message
    const userMessage = messages.filter(msg => msg.role === "user").pop();
    
    if (!userMessage) {
      console.log("No user message found");
      return NextResponse.json({ error: "No user message found" }, { status: 400 });
    }

    // Extract text content from message (handle both string and array formats)
    let messageText = "";
    if (typeof userMessage.content === "string") {
      messageText = userMessage.content;
    } else if (Array.isArray(userMessage.content)) {
      // Extract text from content array
      messageText = userMessage.content
        .filter(item => item.type === "text")
        .map(item => item.text)
        .join(" ");
    }
    
    if (!messageText || messageText.trim().length === 0) {
      console.log("Empty user message");
      // Early return with helpful message instead of calling backend with empty input
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: "text-delta", textDelta: "Please enter a question to analyze." })}\n\n`)
          );
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: "finish", finishReason: "stop" })}\n\n`)
          );
          controller.close();
        },
      });
      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache, no-transform',
          'Connection': 'keep-alive',
          'X-Accel-Buffering': 'no',
        },
      });
    }

    console.log("Processing user message:", messageText);
    
    // Call your Python Hierarchical RAG backend
    const ragBackendUrl = process.env.RAG_BACKEND_URL || "http://localhost:8001";
    console.log("Using Hierarchical RAG backend URL:", ragBackendUrl);
    
    try {
      const response = await fetch(`${ragBackendUrl}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          // Support hierarchical RAG payload format
          query: messageText,
          question: messageText,
          level1_limit: 2,
          level2_limit: 3,
          show_raw_results: false,
          conversation_history: messages.slice(0, -1).map(msg => ({
            role: msg.role,
            content: typeof msg.content === "string" ? msg.content : 
              Array.isArray(msg.content) ? 
                msg.content.filter(item => item.type === "text").map(item => item.text).join(" ") : 
                ""
          }))
        }),
      });

      if (!response.ok) {
        console.error(`Hierarchical RAG backend error: ${response.status} ${response.statusText}`);
        const errorText = await response.text();
        console.error("Hierarchical RAG backend error details:", errorText);
        throw new Error(`Hierarchical RAG backend responded with status: ${response.status}`);
      }

      const ragResult = await response.json();
      console.log("Hierarchical RAG response received:", ragResult);
      
      // Extract response and additional info from hierarchical RAG
      let responseText = ragResult.response || ragResult.answer || "I'm sorry, I couldn't process your request.";
      let searchResults = ragResult.search_results || ragResult.level2_chunks || null;
      let imagesAvailable = ragResult.images_available || null;
      let addresses = ragResult.addresses || null;
      let structured = ragResult.structured || null;
      const normalizeImagePath = (p: string) => {
        if (!p) return p;
        if (p.startsWith('http://') || p.startsWith('https://')) return p;
        const base = new URL(ragBackendUrl);
        if (p.startsWith('extracted_images/')) return `${base.origin}/images/${p.replace('extracted_images/', '')}`;
        if (p.startsWith('/extracted_images/')) return `${base.origin}/images/${p.replace('/extracted_images/', '')}`;
        if (p.startsWith('/')) return `${base.origin}${p}`;
        return `${base.origin}/${p}`;
      };
      
      // Parse JSON response if it's a string (from improved hierarchical RAG)
      if (typeof responseText === 'string' && responseText.trim().startsWith('{')) {
        try {
          const parsedResponse = JSON.parse(responseText);
          responseText = parsedResponse.answer || responseText;
          
          // Extract images_available from the JSON response
          if (parsedResponse.images_available) {
            imagesAvailable = typeof parsedResponse.images_available === 'string' 
              ? JSON.parse(parsedResponse.images_available) 
              : parsedResponse.images_available;
          }
          if (!addresses && parsedResponse.addresses) {
            addresses = parsedResponse.addresses;
          }
          if (!structured && parsedResponse.structured) {
            structured = parsedResponse.structured;
          }
          
          // Extract search results if not already available
          if (!searchResults && parsedResponse.search_results) {
            searchResults = parsedResponse.search_results;
          }
        } catch (parseError) {
          console.log("Response is not JSON, treating as plain text");
        }
      }
      
      // Add hierarchical search info if available
      if (ragResult.level1_docs || ragResult.level2_chunks) {
        const sourceInfo = [];
        if (ragResult.level1_docs?.length > 0) {
          sourceInfo.push(`\n\n📊 **Documents Found**: ${ragResult.level1_docs.length} relevant properties`);
          ragResult.level1_docs.forEach((doc: any, i: number) => {
            sourceInfo.push(`${i + 1}. ${doc.address || doc.doc_id}`);
          });
        }
        
        if (ragResult.level2_chunks?.length > 0) {
          sourceInfo.push(`\n\n🔍 **Relevant Sections**: Found ${ragResult.level2_chunks.length} matching content chunks`);
        }
        
        if (sourceInfo.length > 0) {
          responseText += sourceInfo.join('\n');
        }
      }
      
      // Return streaming response format expected by Assistant UI
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          // First send the text content
          let index = 0;
          let currentText = "";
          
          const sendNext = () => {
            if (index < responseText.length) {
              const char = responseText[index];
              currentText += char;
              
              // Send the incremental text delta
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify({ 
                  type: "text-delta", 
                  textDelta: char 
                })}\n\n`)
              );
              
              index++;
              setTimeout(sendNext, 30);
            } else {
              // Send search results if available
              if (searchResults && searchResults.length > 0) {
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify({ 
                    type: "search-results", 
                    searchResults: searchResults 
                  })}\n\n`)
                );
              }
              
              // Prefer images_available from backend, else derive from search results
              let imagesPayload = imagesAvailable;
              if ((!imagesPayload || imagesPayload.length === 0) && searchResults && searchResults.length > 0) {
                try {
                  imagesPayload = [];
                  
                  searchResults.filter((r: any) => r.chunk_type === 'image').forEach((r: any) => {
                    // Remove "Diagram" from section names and use clean image names
                    let cleanSection = r.section || 'Unknown Image';
                    cleanSection = cleanSection.replace(' Diagram', '').replace('Diagram', '');
                    
                    // Use the image_title if available, otherwise clean the section name
                    const displayTitle = r.image_title || cleanSection;
                    
                    // Extract all image paths from chunk_text (works for both single and multiple images)
                    const imagePaths: string[] = [];
                    
                    if (r.chunk_text) {
                      const lines = r.chunk_text.split('\n');
                      lines.forEach((line: string) => {
                        // Look for lines containing image paths
                        if (line.includes('extracted_images/')) {
                          // Extract the path - handle different formats
                          const match = line.match(/extracted_images\/[^,\s\]]+\.png/);
                          if (match) {
                            imagePaths.push(match[0]);
                          }
                        }
                        // Also check for image_file: format
                        if (line.includes('image_file:')) {
                          const pathMatch = line.split('image_file:')[1]?.trim();
                          if (pathMatch && pathMatch.includes('extracted_images/')) {
                            imagePaths.push(pathMatch);
                          }
                        }
                      });
                    }
                    
                    // If we found image paths, create entries for each
                    if (imagePaths.length > 0) {
                      imagePaths.forEach(imagePath => {
                        // Extract filename for better titles
                        const filename = imagePath.split('/').pop()?.replace('.png', '') || '';
                        
                        // Create user-friendly titles
                        const titleMappings = {
                          'Lengthsimage': '📏 Length Measurements',
                          'Pitch_Degrees': '📐 Roof Pitch (Degrees)',
                          'Pitch_on_12': '📐 Roof Pitch (Rise over 12)',
                          'Rafters': '🏗️ Rafter Structure',
                          'Azimuth': '🧭 Roof Azimuth/Direction',
                          'Area': '📊 Roof Area Measurements',
                          'Roof_Penetrations': '🔍 Roof Penetrations',
                          'Top_View': '🛰️ Aerial/Top View',
                          'North_Side': '⬆️ North Side View',
                          'South_Side': '⬇️ South Side View',
                          'East_Side': '➡️ East Side View',
                          'West_Side': '⬅️ West Side View',
                          'Cover_Image': '🏠 Cover/Overview Image',
                          'Structure_Summary': '📋 Structure Summary'
                        };
                        
                        const imageTitle = titleMappings[filename as keyof typeof titleMappings] || displayTitle;
                        
                        imagesPayload.push({
                          section: imageTitle,
                          description: r.image_description || `${cleanSection} image`,
                          doc_address: r.doc_address || 'Unknown Address',
                          image_file: imagePath,
                          title: imageTitle
                        });
                      });
                    } else {
                      // Fallback: create entry even without image path (for debugging)
                      imagesPayload.push({
                        section: displayTitle,
                        description: r.image_description || r.chunk_text || 'Property image',
                        doc_address: r.doc_address || 'Unknown Address',
                        image_file: r.image_path || r.image_file || '',
                        title: displayTitle
                      });
                    }
                  });
                } catch (e) {
                  console.error('Error processing image data:', e);
                }
              }

              if (imagesPayload && imagesPayload.length > 0) {
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify({ 
                    type: "images-available", 
                    imagesAvailable: imagesPayload 
                  })}\n\n`)
                );
              }
              
              // Include addresses and structured in the final payload
              const finishPayload: any = { type: "finish", finishReason: "stop" };
              if (addresses) finishPayload.addresses = addresses;
              if (structured) finishPayload.structured = structured;
              console.log("finishPayload", JSON.stringify(finishPayload))
              // Send finish signal
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(finishPayload)}\n\n`)
              );
              controller.close();
            }
          };

          sendNext();
        },
        cancel() {
          // Client disconnected
        },
      });

      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache, no-transform',
          'Connection': 'keep-alive',
          'X-Accel-Buffering': 'no',
        },
      });
     

    } catch (backendError) {
      console.error("Error calling RAG backend:", backendError);
      
      // Fallback response if backend is not available
      const fallbackMessage = "I'm currently unable to connect to the property analysis system. Please ensure the backend service is running.";
      
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ 
              type: "text-delta",
              textDelta: fallbackMessage
            })}\n\n`)
          );
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ 
              type: "finish",
              finishReason: "stop"
            })}\n\n`)
          );
          controller.close();
        },
      });

      return new Response(stream, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    }
    
  } catch (error) {
    console.error("Error in chat API:", error);
    console.error("Error stack:", error instanceof Error ? error.stack : 'No stack trace');
    
    // Return a proper error response
    return NextResponse.json(
      { error: "Internal server error", details: error instanceof Error ? error.message : String(error) },
      { status: 500 }
    );
  }
}