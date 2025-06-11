// import { useState } from 'react'
// import reactLogo from './assets/react.svg'
// import viteLogo from '/vite.svg'
import './App.css'

import { useState } from 'react';
import Navigation from './components/Navigation';
import AdminArea from './components/AdminArea';
////import ReferenceTable from './components/ReferenceTable';
import ChatArea from './components/ChatArea';
import Shell from './components/Shell';


const App: React.FC = () => {
  const [selectedContent, setSelectedContent] = useState<string>('query');

  const getHeadingText = (opt: string): string => {
    switch (opt) {
      case 'admin':
        return 'Admin Area';
      case 'shell':
        return 'Shell';
      default:
        return 'Query';
    }
  }

  const getContent = (opt: string): JSX.Element => {
    switch (opt) {
      case 'admin':
        return <AdminArea />;
      case 'shell':
        return <Shell />;
      default:
        return <ChatArea />;
    }
  }
  return (
    <div className="flex h-screen w-screen bg-gray-100">
      <div className="w-1/5 bg-gray-800 text-white">
        <Navigation onSelect={setSelectedContent} />
      </div>
      <div className="w-4/5 flex flex-col p-4">

        {/* Dynamic Content */}
        <div className="bg-white w-full flex-col flex rounded-lg shadow p-4">
          <h2 className="text-2xl font-bold mb-4 p-3">
            {getHeadingText(selectedContent)}
          </h2>
          {getContent(selectedContent)}
        </div>
      </div>
    </div>
  )
}

export default App;
