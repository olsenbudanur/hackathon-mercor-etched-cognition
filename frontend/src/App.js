import { useState, useEffect } from "react";
import "./App.css";  // Import your custom CSS file

export default function StreamingApp() {
  const [words, setWords] = useState([]);
  const [typing, setTyping] = useState(true);

  // Function to determine the color based on the number
  const getColor = (number) => {
    switch (number) {
      case 1:
        return "text-red-500"; // Red
      case 2:
        return "text-orange-500"; // Orange
      case 3:
        return "text-yellow-500"; // Yellow
      case 4:
        return "text-green-500"; // Green
      case 5:
        return "text-teal-500"; // Teal
      case 6:
        return "text-blue-500"; // Blue
      case 7:
        return "text-indigo-500"; // Indigo
      case 8:
        return "text-purple-500"; // Purple
      case 9:
        return "text-pink-500"; // Pink
      case 10:
        return "text-gray-500"; // Gray
      default:
        return "text-black"; // Default color
    }
  };

  useEffect(() => {
    console.log("Connecting to SSE...");

    const eventSource = new EventSource("http://127.0.0.1:8000/stream");

    eventSource.onopen = () => console.log("SSE Connection Opened");
    eventSource.onerror = (error) => console.error("SSE Error:", error);
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);  // Parse the incoming data
        setWords((prev) => [
          ...prev,
          { word: data.word, number: data.number },
        ]);
        setTyping(true); // Trigger typing effect
      } catch (error) {
        console.error("Error parsing JSON:", error);
      }
    };

    return () => {
      console.log("Closing SSE...");
      eventSource.close();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 flex justify-center items-center p-4">
      <div className="w-full max-w-4xl bg-white rounded-lg p-6 shadow-lg border border-gray-200">
        <h1 className="text-3xl font-medium text-gray-800 mb-6 text-center">
          Streaming Messages
        </h1>
        <div className="message-box">
          <pre className="text-base font-mono text-gray-700 whitespace-pre-wrap overflow-auto break-words h-96 p-4 bg-gray-100 rounded-md">
            {words.map((item, index) => (
              <span key={index} className={`${getColor(item.number)} mr-2`}>
                {item.word}
              </span>
            ))}
            {typing && <span className="typing-effect">|</span>}
          </pre>
        </div>
      </div>
    </div>
  );
}
