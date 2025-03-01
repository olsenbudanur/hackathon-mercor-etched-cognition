import { useState, useEffect } from "react";
import "./App.css";  // Import your custom CSS file

export default function StreamingApp() {
    const [text, setText] = useState("");
  
    useEffect(() => {
      console.log("Connecting to SSE...");
  
      const eventSource = new EventSource("http://127.0.0.1:8000/stream");
  
      eventSource.onopen = () => console.log("SSE Connection Opened");
      eventSource.onerror = (error) => console.error("SSE Error:", error);
      eventSource.onmessage = (event) => {
        console.log("Received:", event.data);
      };
      eventSource.onmessage = (event) => {
        setText((prev) => prev + event.data);
      };
      return () => {
        console.log("Closing SSE...");
        eventSource.close();
      };
    }, []);
  
  return (
    <div className="min-h-screen bg-gray-50 flex justify-center items-center p-4">
        <div className="w-full max-w-4xl bg-white rounded-lg p-6 border border-gray-200">
          <h1 className="text-3xl font-medium text-gray-800 mb-6 text-center">
            Streaming Messages
          </h1>
          <pre className="text-base font-mono text-gray-700 whitespace-pre-wrap overflow-auto h-96 p-4 bg-gray-100 rounded-md">
            {text}
          </pre>
        </div>
      </div>
      );
}
  
  
  