"use client";

import React, { useState, useEffect } from 'react';
import { ThreadPrimitive } from "@assistant-ui/react";
import { Button } from "@/components/ui/button";

interface DocumentStats {
  total_documents: number;
  total_chunks: number;
  level1_entities?: number;
  level2_entities?: number;
  vector_store_type: string;
  status: string;
}

interface SystemStatus {
  status: string;
  documents_loaded: number;
  level1_entities?: number;
  level2_entities?: number;
  vector_store_active?: boolean;
  backend_version: string;
  rag_available?: boolean;
}

export function PropertyDashboard() {
  const [docStats, setDocStats] = useState<DocumentStats | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
    // Refresh stats every 30 seconds
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      // Use public env var in the browser; fall back to server env var and localhost
      const backendUrl =
        process.env.NEXT_PUBLIC_RAG_BACKEND_URL ||
        process.env.RAG_BACKEND_URL ||
        "http://localhost:8001";  // Updated default port for hierarchical RAG
      
      const [statsResponse, healthResponse] = await Promise.all([
        fetch(`${backendUrl}/documents/stats`),
        fetch(`${backendUrl}/health`)
      ]);

      if (statsResponse.ok) {
        const stats = await statsResponse.json();
        setDocStats(stats);
      }

      if (healthResponse.ok) {
        const health = await healthResponse.json();
        setSystemStatus(health);
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
      case 'active':
        return 'bg-green-500';
      case 'no_documents':
      case 'not_initialized':
        return 'bg-yellow-500';
      default:
        return 'bg-red-500';
    }
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* System Status */}
      <div>
        <h3 className="mb-3 text-sm font-medium text-foreground">
          System Status
        </h3>
        <div className="space-y-2">
          {systemStatus && (
            <>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Backend</span>
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${getStatusColor(systemStatus.status)}`} />
                  <span className="text-xs capitalize">{systemStatus.status}</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">RAG System</span>
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${systemStatus.rag_available ? 'bg-green-500' : 'bg-red-500'}`} />
                  <span className="text-xs">{systemStatus.rag_available ? 'Active' : 'Inactive'}</span>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Vector Store</span>
                <div className="flex items-center gap-2">
                  <div className={`h-2 w-2 rounded-full ${'bg-green-500'}`} />
                  <span className="text-xs">{'Active'}</span>
                </div>
              </div>
              {/* <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Documents Loaded</span>
                <span className="text-xs font-mono">{systemStatus.documents_loaded}</span>
              </div> */}
            </>
          )}
        </div>
      </div>

      {/* Document Statistics */}
      <div>
        <h3 className="mb-3 text-sm font-medium text-foreground">
          Hierarchical Index Status
        </h3>
        <div className="space-y-2">
          {docStats && (
            <>
              {/* <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Total Documents</span>
                <span className="text-xs font-mono">{docStats.total_documents}</span>
              </div> */}
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Level 3 (DDD Diagrams)</span>
                <span className="text-xs font-mono">{22}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Level 2 (Chunks)</span>
                <span className="text-xs font-mono">{docStats.level2_entities || docStats.total_chunks}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">Store Type</span>
                <span className="text-xs font-mono capitalize">{docStats.vector_store_type}</span>
              </div>
            </>
          )}
        </div>
      </div>


      {/* Quick Chat Prompts */}
      <div>
        <h3 className="mb-3 text-sm font-medium text-foreground">
          Quick Chat Prompts
        </h3>
        <div className="space-y-2">
          {[
            { icon: "💬", label: "Ask about roof conditions", query: "What are the current roof conditions across all properties?" },
            { icon: "📋", label: "Request measurement summary", query: "Give me a summary of all roof measurements for property ID 67668772" },
            { icon: "🔍", label: "Find specific property details", query: "Show me detailed information about property with roof report number 44995431" },
            { icon: "📊", label: "Compare roof areas", query: "Compare the roof areas between different properties in the database" },
            { icon: "🖼️", label: "Describe property images", query: "Describe what you can see in the images for property 67676202" }
          ].map((item, index) => (
            <ThreadPrimitive.Suggestion key={index} prompt={item.query} method="replace" autoSend asChild>
              <button
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-left text-xs text-foreground hover:bg-accent hover:text-accent-foreground flex items-center gap-2"
                aria-label={`Quick prompt: ${item.label}`}
              >
                <span>{item.icon}</span>
                <span>{item.label}</span>
              </button>
            </ThreadPrimitive.Suggestion>
          ))}
        </div>
      </div>
    </div>
  );
}