"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ImageGalleryButton } from "./ImageGalleryButton";

interface MeasurementProperty {
  property_id: string;
  property_address: string;
  measurements: any;
  images: {
    [key: string]: string;
  };
}

interface MeasurementData {
  status: string;
  measurement_type: string;
  properties_found: number;
  measurement_data: MeasurementProperty[];
}

interface MeasurementViewerProps {
  measurementType: string;
}

export function MeasurementViewer({ measurementType }: MeasurementViewerProps) {
  const [measurementData, setMeasurementData] = useState<MeasurementData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedView, setSelectedView] = useState<'table' | 'cards'>('cards');

  useEffect(() => {
    // First try to load data from sessionStorage (from chat button)
    const storageKeys = {
      'lengths': 'lengthsData',
      'rafters': 'raftersData',
      'area': 'areaData',
      'azimuth': 'azimuthData'
    };
    
    const storageKey = storageKeys[measurementType as keyof typeof storageKeys];
    const sessionData = sessionStorage.getItem(storageKey);
    
    if (sessionData) {
      try {
        const parsedData = JSON.parse(sessionData);
        setMeasurementData({
          status: 'success',
          measurement_type: measurementType,
          properties_found: parsedData.length,
          measurement_data: parsedData
        });
        setLoading(false);
        return;
      } catch (err) {
        console.error('Error parsing session data:', err);
      }
    }
    
    // Fallback to API if no session data
    fetchMeasurementData();
  }, [measurementType]);

  const fetchMeasurementData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const backendUrl =
        process.env.NEXT_PUBLIC_RAG_BACKEND_URL ||
        process.env.RAG_BACKEND_URL ||
        "http://localhost:8001";
      
      const response = await fetch(`${backendUrl}/${measurementType}-data`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setMeasurementData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      console.error('Error fetching measurement data:', err);
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

  const renderMeasurementValue = (key: string, value: any) => {
    if (!value || value === "0'" || value === "0' (0 Lengths)") return null;
    
    const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    
    return (
      <div key={key} className="flex justify-between py-1">
        <span className="text-sm text-muted-foreground">{displayKey}:</span>
        <span className="text-sm font-medium">{String(value)}</span>
      </div>
    );
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
          <h2 className="text-lg font-semibold text-foreground mb-4">{measurementType.charAt(0).toUpperCase() + measurementType.slice(1)} Measurement Information</h2>
          <div className="border border-red-200 bg-red-50 p-4 rounded-md max-w-md">
            <p className="text-red-700">Error loading measurement data: {error}</p>
            <Button onClick={fetchMeasurementData} className="mt-2" variant="outline" size="sm">
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  if (!measurementData || measurementData.properties_found === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-lg font-semibold text-foreground mb-4">{measurementType.charAt(0).toUpperCase() + measurementType.slice(1)} Measurement Information</h2>
          <p className="text-muted-foreground">No {measurementType} measurement data found.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b bg-background flex-shrink-0">
        <div>
          <h2 className="text-lg font-semibold text-foreground">{measurementType.charAt(0).toUpperCase() + measurementType.slice(1)} Measurement Information</h2>
          <p className="text-sm text-muted-foreground">
            Found {measurementData.properties_found} properties with {measurementType} measurement data
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
          <div className="grid gap-6">
            {measurementData.measurement_data.map((property) => (
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
                  </div>
                </CardHeader>

                <CardContent className="space-y-6">
                  {/* Measurement Details */}
                  <div>
                    <h4 className="text-sm font-medium mb-3">Measurement Details:</h4>
                    <div className="bg-muted/50 rounded-md p-3 space-y-1">
                      {Object.entries(property.measurements).map(([key, value]) => {
                        if (key === 'report_id' || key === 'property_address' || key === 'date') return null;
                        return renderMeasurementValue(key, value);
                      })}
                    </div>
                  </div>

                  {/* Images */}
                  {Object.keys(property.images).length > 0 && (
                    <div className="space-y-4">
                      <h4 className="text-sm font-medium">Measurement Diagrams:</h4>
                      {Object.entries(property.images).map(([imageType, imagePath]) => (
                        <div key={imageType}>
                          <h5 className="text-sm font-medium mb-2 capitalize">{imageType.replace(/_/g, ' ')}:</h5>
                          <div className="border rounded-md overflow-hidden">
                            <img
                              src={getImageUrl(imagePath)}
                              alt={`${imageType} diagram`}
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
                      ))}
                    </div>
                  )}
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
                  <th className="px-3 py-2 text-left">Key Measurements</th>
                  <th className="px-3 py-2 text-left">Images</th>
                </tr>
              </thead>
              <tbody>
                {measurementData.measurement_data.map((property) => (
                  <tr key={property.property_id} className="border-t">
                    <td className="px-3 py-2 font-medium">{property.property_id}</td>
                    <td className="px-3 py-2">{property.property_address}</td>
                    <td className="px-3 py-2">
                      <div className="space-y-1">
                        {Object.entries(property.measurements).slice(0, 3).map(([key, value]) => {
                          if (key === 'report_id' || key === 'property_address' || key === 'date' || !value) return null;
                          return (
                            <div key={key} className="text-xs">
                              <span className="font-medium">{key.replace(/_/g, ' ')}:</span> {String(value)}
                            </div>
                          );
                        })}
                      </div>
                    </td>
                    <td className="px-3 py-2">
                      <div className="flex gap-1">
                        {Object.keys(property.images).map((imageType) => (
                          <Badge key={imageType} variant="outline" className="text-xs">
                            {imageType.split('_')[0]}
                          </Badge>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        
        {/* Image Gallery Button */}
        {measurementData && (
          <div className="mt-6 flex justify-center">
            <ImageGalleryButton imageData={measurementData.measurement_data} />
          </div>
        )}
      </div>
    </div>
  );
}