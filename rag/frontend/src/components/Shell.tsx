import { useState, useRef, useEffect } from 'react';

const Shell: React.FC = () => {
  const [inputCmd, setInputCmd] = useState<string>('');
  const [outputText, setOutputText] = useState<string>('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = async () => {
    if (!inputCmd.trim()) return;

    try {
      const response = await fetch('/shell', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cmd: inputCmd }),
      });
      const data = await response.json();
      const { cmd, output, error } = data;
      const newOutput = `\n# ${cmd}\n${error || output}\n`;
      setOutputText((prev) => prev + newOutput);
      setInputCmd('');
    } catch (err) {
      const newOutput = `\n# ${inputCmd}\nError: ${err.message}\n`;
      setOutputText((prev) => prev + newOutput);
      setInputCmd('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Scroll to bottom on output change
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.scrollTop = textareaRef.current.scrollHeight;
    }
  }, [outputText]);

  return (
    <div className="flex-1 flex flex-col">
      <textarea
        ref={textareaRef}
        className="w-full h-[80vh] border p-2 mb-4 rounded bg-gray-50 font-mono text-sm resize-none overflow-y-auto"
        value={outputText}
        readOnly
        placeholder="Command output will appear here..."
      />
      <div className="flex">
        <input
          type="text"
          className="flex-1 border p-2 rounded-l font-mono text-sm"
          value={inputCmd}
          onChange={(e) => setInputCmd(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter command..."
        />
        <button
          onClick={handleSubmit}
          className="bg-blue-500 text-white p-2 rounded-r hover:bg-blue-600"
        >
          Run
        </button>
      </div>
    </div>
  );
};

export default Shell;
