"use client";

import React from 'react';
import { Button } from "@/components/ui/button";
import { BarChart3, ExternalLink } from "lucide-react";
import { useRouter } from "next/navigation";

interface RoofPitchResultButtonProps {
  roofPitchData: any[];
  propertiesCount: number;
}

export function RoofPitchResultButton({ roofPitchData, propertiesCount }: RoofPitchResultButtonProps) {
  const router = useRouter();

  const handleViewResults = () => {
    // Store the roof pitch data in sessionStorage so the results page can access it
    sessionStorage.setItem('roofPitchData', JSON.stringify(roofPitchData));
    router.push('/roof-pitch-results');
  };

  return (
    <div className="mt-4 p-4 border border-border rounded-lg bg-card/50">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <BarChart3 className="h-5 w-5 text-primary" />
          <div>
            <h4 className="font-medium text-sm">Roof Pitch Analysis Ready</h4>
            <p className="text-xs text-muted-foreground">
              Found {propertiesCount} properties with detailed pitch data and diagrams
            </p>
          </div>
        </div>
        <Button 
          onClick={handleViewResults}
          variant="default" 
          size="sm"
          className="flex items-center gap-2"
        >
          <ExternalLink className="h-4 w-4" />
          View Results
        </Button>
      </div>
    </div>
  );
}