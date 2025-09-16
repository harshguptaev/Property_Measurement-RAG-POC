"use client";

import { AssistantRuntimeProvider, useLocalRuntime } from "@assistant-ui/react";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import { Thread } from "@/components/assistant-ui/thread";
import { PropertyDashboard } from "@/components/PropertyDashboard";

export default function PropertyAnalysisApp() {
  const runtime = useLocalRuntime({
    async *run({ messages }) {
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ messages }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("No response body");
        }

        const decoder = new TextDecoder();
        let buffer = "";
        let accumulatedText = ""; // Accumulate text deltas

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (line.startsWith("data: ")) {
                try {
                  const data = JSON.parse(line.slice(6));
                  
                  if (data.type === "text-delta") {
                    accumulatedText += data.textDelta; // Accumulate the text
                    yield {
                      content: [{ type: "text", text: accumulatedText }],
                    };
                  } else if (data.type === "finish") {
                    return;
                  }
                } catch (parseError) {
                  // Silently handle parse errors
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }
      } catch (error) {
        // Yield an error message to the user
        yield {
          content: [{ type: "text", text: `Error: ${error instanceof Error ? error.message : String(error)}` }],
        };
      }
    },
  });

  return (
    <div className="h-screen bg-background">
      <AssistantRuntimeProvider runtime={runtime}>
        {/* Header */}
        <div className="border-b border-border bg-card px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-foreground">
                🏠 Property Measurement RAG System
              </h1>
              <p className="text-sm text-muted-foreground">
                AI-powered property document analysis and question answering
              </p>
            </div>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-green-500" />
                <span>Connected</span>
              </div>
            </div>
          </div>
        </div>

        {/* Main content area */}
        <div className="flex h-[calc(100vh-80px)]">
          {/* Sidebar with thread list */}
          <div className="w-80 border-r border-border bg-card/50">
            <div className="p-4">
              <h2 className="mb-4 text-sm font-medium text-foreground">
                Conversation History
              </h2>
              <ThreadList />
            </div>
          </div>

          {/* Main chat area */}
          <div className="flex-1">
            <Thread />
          </div>

          {/* Property Dashboard Sidebar */}
          <div className="w-80 border-l border-border bg-card/30 p-4">
            <PropertyDashboard />
          </div>
        </div>
      </AssistantRuntimeProvider>
    </div>
  );
}
