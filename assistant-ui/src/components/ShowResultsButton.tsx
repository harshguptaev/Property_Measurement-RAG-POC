"use client";

import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ImageIcon, FileTextIcon, MapPinIcon, RulerIcon } from "lucide-react";

interface ImageData {
  section: string;
  description: string;
  image_file?: string;
  image_files?: string[];
  doc_address: string;
  title?: string;
  image_path?: string;
  image_title?: string;
  image_description?: string;
}

interface ShowResultsButtonProps {
  imagesAvailable?: ImageData[];
  searchResults?: any[];
  className?: string;
}

export function ShowResultsButton({ 
  imagesAvailable = [], 
  searchResults = [], 
  className = "" 
}: ShowResultsButtonProps) {
  const [isOpen, setIsOpen] = useState(false);

  const resolveSrc = (p?: string) => {
    if (!p) return p as any;
    if (p.startsWith('http://') || p.startsWith('https://')) return p;
    if (p.startsWith('extracted_images/')) {
      return `http://localhost:8001/images/${p.replace('extracted_images/', '')}`;
    }
    if (p.startsWith('/')) return p;
    return `/${p}`;
  };

  const totalImages = imagesAvailable.length;

  const totalResults = searchResults.length;

  if (totalImages === 0 && totalResults === 0) {
    return null;
  }
  console.log("imagesAvailable", imagesAvailable)
  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button 
          variant="outline" 
          className={`mt-3 ${className}`}
          size="sm"
        >
          <ImageIcon className="w-4 h-4 mr-2" />
          Show Results ({totalImages} images, {totalResults} chunks)
        </Button>
      </DialogTrigger>
      
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ImageIcon className="w-5 h-5" />
            Retrieved Results & Images
          </DialogTitle>
        </DialogHeader>
        
        <Tabs defaultValue="images" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="images" className="flex items-center gap-2">
              <ImageIcon className="w-4 h-4" />
              Images ({totalImages})
            </TabsTrigger>
            <TabsTrigger value="chunks" className="flex items-center gap-2">
              <FileTextIcon className="w-4 h-4" />
              Search Results ({totalResults})
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="images" className="mt-4 max-h-[70vh] overflow-y-auto">
            {imagesAvailable && imagesAvailable.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {imagesAvailable.map((item, index) => (
                  <div key={index} className="space-y-2">
                    <img 
                      src={resolveSrc(item.image_file || item.image_path)} 
                      alt={item.title || item.section}
                      className="w-full h-48 object-cover rounded-md border"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.src = '/placeholder-image.png';
                      }}
                    />
                    <div className="text-xs">
                      <p className="font-medium">{item.title || item.section}</p>
                      <p className="text-muted-foreground">{item.doc_address}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <ImageIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No images available in the search results</p>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="chunks" className="mt-4 max-h-[70vh] overflow-y-auto">
            {searchResults.length > 0 ? (
              <div className="space-y-4">
                {searchResults.map((result, index) => (
                  <Card key={index}>
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="text-sm font-medium">
                            {result.section || 'Unknown Section'}
                          </CardTitle>
                          <div className="text-xs mt-1 flex items-center gap-2 text-muted-foreground">
                            <MapPinIcon className="w-3 h-3" />
                            <span>{result.doc_address || result.address || 'Unknown Address'}</span>
                            <Badge variant="outline" className="text-xs ml-2">
                              {result.chunk_type || 'text'}
                            </Badge>
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge variant="secondary" className="text-xs">
                            Score: {result.distance ? (100 - result.distance).toFixed(1) : 'N/A'}
                          </Badge>
                        </div>
                      </div>
                    </CardHeader>
                    
                    <CardContent className="pt-0">
                      <div className="text-sm text-muted-foreground">
                        {result.chunk_text && (
                          <div className="bg-muted/50 p-3 rounded-md">
                            <pre className="whitespace-pre-wrap text-xs font-mono">
                              {result.chunk_text.length > 300 
                                ? `${result.chunk_text.substring(0, 300)}...` 
                                : result.chunk_text
                              }
                            </pre>
                          </div>
                        )}
                        
                        {result.chunk_type === 'image' && (
                          <div className="mt-2 flex items-center gap-2 text-xs">
                            <ImageIcon className="w-4 h-4" />
                            <span>Image chunk - see Images tab for visual content</span>
                          </div>
                        )}
                        
                        <div className="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
                          <span>ID: {result.chunk_id || 'N/A'}</span>
                          <span>Doc: {result.doc_id || 'N/A'}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <FileTextIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No search results available</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
