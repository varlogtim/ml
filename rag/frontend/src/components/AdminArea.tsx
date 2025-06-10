import { useState, useEffect } from 'react';

const AdminArea: React.FC = () => {
  const [configText, setConfigText] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(true);

  // Fetch config on mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch('/config');
        if (!response.ok) throw new Error('Failed to fetch config');
        const data = await response.json();
        setConfigText(data.config || '{}');
      } catch (error) {
        setErrorMessage('Error fetching config: ' + (error as Error).message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchConfig();
  }, []);

  // Handle Configure button click
  const handleConfigure = async () => {
    setErrorMessage(''); // Clear previous errors
    try {
      const response = await fetch('/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: configText }),
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Unknown error');
      }
      setErrorMessage('Configuration saved successfully!');
    } catch (error) {
      setErrorMessage('Error: ' + (error as Error).message);
    }
  };

  return (
    <div className="h-full flex flex-col p-4 bg-white rounded-lg shadow">
      <h2 className="text-1xl font-bold mb-4">Server Configuration</h2>
      {isLoading ? (
        <p>Loading...</p>
      ) : (
        <div className="flex-1 flex flex-col">
          <textarea
            className="h-50 border p-2 mb-4 rounded resize-y bg-gray-50 font-mono text-sm"
            value={configText}
            onChange={(e) => setConfigText(e.target.value)}
            placeholder="Loading JSON Config..."
          />
          {errorMessage && (
            <p
              className={`mb-4 p-2 rounded ${
                errorMessage.includes('successfully')
                  ? 'bg-green-100 text-green-700'
                  : 'bg-red-100 text-red-700'
              }`}
            >
              {errorMessage}
            </p>
          )}
          <button
            onClick={handleConfigure}
            className="bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
          >
            Configure
          </button>
        </div>
      )}
    </div>
  );
};

export default AdminArea;
