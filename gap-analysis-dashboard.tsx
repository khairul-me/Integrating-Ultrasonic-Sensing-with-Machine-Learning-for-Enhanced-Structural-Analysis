import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import Papa from 'papaparse';

const GapAnalysisDashboard = () => {
  const [data, setData] = useState([]);
  
  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await window.fs.readFile('gap_scan_20241122_210743.csv', { encoding: 'utf8' });
        
        Papa.parse(response, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            setData(results.data.map(row => ({
              ...row,
              angle: parseFloat(row.angle),
              confidence: parseFloat(row.confidence),
              precision: parseFloat(row.precision),
              recall: parseFloat(row.recall),
              f1_score: parseFloat(row.f1_score)
            })));
          }
        });
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };
    
    loadData();
  }, []);

  return (
    <div className="w-full space-y-4 p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Detection Performance Over Time</CardTitle>
          </CardHeader>
          <CardContent className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="angle" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="precision" stroke="#8884d8" name="Precision" />
                <Line type="monotone" dataKey="recall" stroke="#82ca9d" name="Recall" />
                <Line type="monotone" dataKey="f1_score" stroke="#ffc658" name="F1 Score" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Confidence and Distance Analysis</CardTitle>
          </CardHeader>
          <CardContent className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="angle" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="confidence" stroke="#ff7300" name="Confidence" />
                <Line type="monotone" dataKey="filtered_distance" stroke="#387908" name="Filtered Distance" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default GapAnalysisDashboard;