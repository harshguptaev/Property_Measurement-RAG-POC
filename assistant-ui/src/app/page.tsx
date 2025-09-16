"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import { Thread } from "@/components/assistant-ui/thread";
import { PropertyDashboard } from "@/components/PropertyDashboard";

export default function PropertyAnalysisApp() {
  const runtime = useChatRuntime();

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
