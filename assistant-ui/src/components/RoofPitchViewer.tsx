"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface PitchBreakdown {
  pitch: string;
  area: string;
  percentage: string;
}

interface RoofPitchProperty {
  property_id: string;
  property_address: string;
  predominant_pitch: string;
  total_roof_facets: string;
  total_roof_area: string;
  pitch_breakdown: PitchBreakdown[];
  images: {
    pitch_on_12?: string;
    pitch_degrees?: string;
  };
}

interface RoofPitchData {
  status: string;
  properties_found: number;
  roof_pitch_data: RoofPitchProperty[];
}

export function RoofPitchViewer() {
  const [roofData, setRoofData] = useState<RoofPitchData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<'table' | 'cards'>('cards');

  useEffect(() => {
    // First try to load data from sessionStorage (from chat button)
    const sessionData = sessionStorage.getItem('roofPitchData');
    if (sessionData) {
      try {
        const parsedData = JSON.parse(sessionData);
        setRoofData({
          status: 'success',
          properties_found: parsedData.length,
          roof_pitch_data: parsedData
        });
        setLoading(false);
        return;
      } catch (err) {
        console.error('Error parsing session data:', err);
      }
    }
    
    // Fallback to API if no session data
    fetchRoofPitchData();
  }, []);

  const fetchRoofPitchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const backendUrl =
        process.env.NEXT_PUBLIC_RAG_BACKEND_URL ||
        process.env.RAG_BACKEND_URL ||
        "http://localhost:8001";
      
      const response = await fetch(`${backendUrl}/roof-pitch-data`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setRoofData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error fetching roof pitch data:', err);
    } finally {
      setLoading(false);
    }
  };

  const getImageUrl = (imagePath: string) => {
    const backendUrl =
      process.env.NEXT_PUBLIC_RAG_BACKEND_URL ||
      process.env.RAG_BACKEND_URL ||
      "http://localhost:8001";
    
    // Remove 'extracted_images/' prefix if present
    const cleanPath = imagePath.replace(/^extracted_images\//, '');
    return `${backendUrl}/images/${cleanPath}`;
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="animate-pulse text-center">
          <div className="h-6 bg-gray-200 rounded w-64 mb-4 mx-auto"></div>
          <div className="space-y-2">
            <div className="h-32 bg-gray-200 rounded max-w-md mx-auto"></div>
            <div className="h-32 bg-gray-200 rounded max-w-md mx-auto"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-lg font-semibold text-foreground mb-4">Roof Pitch Information</h2>
          <div className="border border-red-200 bg-red-50 p-4 rounded-md max-w-md">
            <p className="text-red-700">Error loading roof pitch data: {error}</p>
            <Button onClick={fetchRoofPitchData} className="mt-2" variant="outline" size="sm">
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  if (!roofData || roofData.properties_found === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-lg font-semibold text-foreground mb-4">Roof Pitch Information</h2>
          <p className="text-muted-foreground">No roof pitch data found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b bg-background flex-shrink-0">
        <div>
          <h2 className="text-lg font-semibold text-foreground">Roof Pitch Information</h2>
          <p className="text-sm text-muted-foreground">
            Found {roofData.properties_found} properties with roof pitch data
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => setSelectedView('cards')}
            variant={selectedView === 'cards' ? 'default' : 'outline'}
            size="sm"
          >
            Cards
          </Button>
          <Button
            onClick={() => setSelectedView('table')}
            variant={selectedView === 'table' ? 'default' : 'outline'}
            size="sm"
          >
            Table
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4">
        {selectedView === 'cards' ? (
          <div className="grid gap-6">{roofData.roof_pitch_data.map((property) => (
            <Card key={property.property_id} className="w-full">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-base">
                      Property {property.property_id}
                    </CardTitle>
                    <p className="text-sm text-muted-foreground mt-1">
                      {property.property_address}
                    </p>
                  </div>
                  <Badge variant="secondary">
                    {property.predominant_pitch || 'N/A'}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Key metrics */}
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Predominant Pitch:</span>
                    <p className="font-medium">{property.predominant_pitch || 'N/A'}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Total Roof Facets:</span>
                    <p className="font-medium">{property.total_roof_facets || 'N/A'}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Total Roof Area:</span>
                    <p className="font-medium">{property.total_roof_area || 'N/A'}</p>
                  </div>
                </div>

                {/* Pitch breakdown table */}
                {property.pitch_breakdown && property.pitch_breakdown.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium mb-2">Pitch Breakdown:</h4>
                    <div className="border rounded-md overflow-hidden">
                      <table className="w-full text-sm">
                        <thead className="bg-muted">
                          <tr>
                            <th className="px-3 py-2 text-left">Pitch</th>
                            <th className="px-3 py-2 text-left">Area</th>
                            <th className="px-3 py-2 text-left">% of Roof</th>
                          </tr>
                        </thead>
                        <tbody>
                          {property.pitch_breakdown.map((pitch, index) => (
                            <tr key={index} className="border-t">
                              <td className="px-3 py-2 font-medium">{pitch.pitch}</td>
                              <td className="px-3 py-2">{pitch.area}</td>
                              <td className="px-3 py-2">{pitch.percentage}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Images */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {property.images.pitch_on_12 && (
                    <div>
                      <h4 className="text-sm font-medium mb-2">Pitch (on 12) Diagram:</h4>
                      <div className="border rounded-md overflow-hidden">
                        <img
                          src={getImageUrl(property.images.pitch_on_12)}
                          alt="Roof Pitch on 12 Diagram"
                          className="w-full h-auto"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.style.display = 'none';
                            target.nextElementSibling?.classList.remove('hidden');
                          }}
                        />
                        <div className="hidden p-4 text-center text-muted-foreground text-sm">
                          Image not available
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {property.images.pitch_degrees && (
                    <div>
                      <h4 className="text-sm font-medium mb-2">Pitch (Degrees) Diagram:</h4>
                      <div className="border rounded-md overflow-hidden">
                        <img
                          src={getImageUrl(property.images.pitch_degrees)}
                          alt="Roof Pitch Degrees Diagram"
                          className="w-full h-auto"
                          onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.style.display = 'none';
                            target.nextElementSibling?.classList.remove('hidden');
                          }}
                        />
                        <div className="hidden p-4 text-center text-muted-foreground text-sm">
                          Image not available
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="border rounded-md overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted">
              <tr>
                <th className="px-3 py-2 text-left">Property ID</th>
                <th className="px-3 py-2 text-left">Address</th>
                <th className="px-3 py-2 text-left">Predominant Pitch</th>
                <th className="px-3 py-2 text-left">Roof Facets</th>
                <th className="px-3 py-2 text-left">Total Area</th>
                <th className="px-3 py-2 text-left">Images</th>
              </tr>
            </thead>
            <tbody>
              {roofData.roof_pitch_data.map((property) => (
                <tr key={property.property_id} className="border-t">
                  <td className="px-3 py-2 font-medium">{property.property_id}</td>
                  <td className="px-3 py-2">{property.property_address}</td>
                  <td className="px-3 py-2">
                    <Badge variant="secondary">{property.predominant_pitch || 'N/A'}</Badge>
                  </td>
                  <td className="px-3 py-2">{property.total_roof_facets || 'N/A'}</td>
                  <td className="px-3 py-2">{property.total_roof_area || 'N/A'}</td>
                  <td className="px-3 py-2">
                    <div className="flex gap-1">
                      {property.images.pitch_on_12 && (
                        <Badge variant="outline" className="text-xs">12</Badge>
                      )}
                      {property.images.pitch_degrees && (
                        <Badge variant="outline" className="text-xs">°</Badge>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      </div>
    </div>
  );
}