"use client";

import React from 'react';
import { RoofPitchViewer } from '@/components/RoofPitchViewer';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import { useRouter } from 'next/navigation';

export default function RoofPitchResultsPage() {
  const router = useRouter();

  return (
    <div className="h-screen overflow-hidden flex flex-col">
      {/* Header */}
      <div className="bg-background border-b border-border p-4 flex items-center gap-4 flex-shrink-0">
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={() => router.back()}
          className="flex items-center gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Chat
        </Button>
        <div>
          <h1 className="text-lg font-semibold">Roof Pitch Analysis Results</h1>
          <p className="text-sm text-muted-foreground">Detailed property roof pitch measurements and diagrams</p>
        </div>
      </div>

      {/* Main Content - Non-scrollable container */}
      <div className="flex-1 overflow-hidden">
        <RoofPitchViewer />
      </div>
    </div>
  );
}