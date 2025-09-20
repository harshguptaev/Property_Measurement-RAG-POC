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
    
    // Call your Python RAG backend
    const ragBackendUrl = process.env.RAG_BACKEND_URL || "http://localhost:8000";
    console.log("Using RAG backend URL:", ragBackendUrl);
    
    try {
      const response = await fetch(`${ragBackendUrl}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          // Support both backend payloads: some expect `query`, others expect `question`
          query: messageText,
          question: messageText,
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
        console.error(`RAG backend error: ${response.status} ${response.statusText}`);
        const errorText = await response.text();
        console.error("RAG backend error details:", errorText);
        throw new Error(`RAG backend responded with status: ${response.status}`);
      }

      const ragResult = await response.json();
      console.log("RAG response received:", ragResult);
      
      // Return streaming response format expected by Assistant UI
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          const responseText = ragResult.response || ragResult.answer || "I'm sorry, I couldn't process your request.";
          
          // Stream the response character by character for smooth display
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
              setTimeout(sendNext, 30); // Faster streaming for smoother experience
            } else {
              // Send finish signal
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify({ 
                  type: "finish", 
                  finishReason: "stop" 
                })}\n\n`)
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