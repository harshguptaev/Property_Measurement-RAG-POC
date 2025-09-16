import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 30;

interface PropertyRAGMessage {
  role: "user" | "assistant";
  content: string;
}

export async function POST(req: NextRequest) {
  try {
    const { messages }: { messages: PropertyRAGMessage[] } = await req.json();
    
    // Get the latest user message
    const userMessage = messages.filter(msg => msg.role === "user").pop();
    
    if (!userMessage) {
      return NextResponse.json({ error: "No user message found" }, { status: 400 });
    }
    if (!userMessage.content || userMessage.content.trim().length === 0) {
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

    // Call your Python RAG backend
    const ragBackendUrl = process.env.RAG_BACKEND_URL || "http://localhost:8000";
    
    try {
      const response = await fetch(`${ragBackendUrl}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          // Support both backend payloads: some expect `query`, others expect `question`
          query: userMessage.content,
          question: userMessage.content,
          conversation_history: messages.slice(0, -1).map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        }),
      });

      if (!response.ok) {
        throw new Error(`RAG backend responded with status: ${response.status}`);
      }

      const ragResult = await response.json();
      
      // Return streaming response format expected by Assistant UI
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        start(controller) {
          const responseText = ragResult.response || ragResult.answer || "I'm sorry, I couldn't process your request.";
          const chunks = responseText.split(' ');
          let index = 0;
          let closed = false;
          const timers = new Set<any>();

          const safeEnqueue = (data: string) => {
            if (closed) return;
            try {
              controller.enqueue(encoder.encode(data));
            } catch {
              closed = true;
              // best-effort cleanup
              for (const t of timers) clearTimeout(t);
              timers.clear();
            }
          };

          const schedule = (fn: () => void, delay = 50) => {
            if (closed) return;
            const t = setTimeout(fn, delay);
            timers.add(t);
          };

          const finish = () => {
            if (closed) return;
            safeEnqueue(`data: ${JSON.stringify({ type: "finish", finishReason: "stop" })}\n\n`);
            closed = true;
            for (const t of timers) clearTimeout(t);
            timers.clear();
            try { controller.close(); } catch { /* noop */ }
          };

          const sendNext = () => {
            if (closed) return;
            if (index < chunks.length) {
              const chunk = chunks[index] + (index < chunks.length - 1 ? ' ' : '');
              safeEnqueue(`data: ${JSON.stringify({ type: "text-delta", textDelta: chunk })}\n\n`);
              index++;
              schedule(sendNext, 50);
            } else {
              finish();
            }
          };

          sendNext();
        },
        cancel() {
          // Client disconnected; stop scheduling and prevent further writes
          // Note: controller is not available here; just mark closed and clear timers
          // Types deliberately loose to support both Node and Edge runtimes
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const noop = null;
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
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}