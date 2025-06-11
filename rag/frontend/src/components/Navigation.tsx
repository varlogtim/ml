interface NavigationProps {
  onSelect: (content: string) => void;
}

const Navigation: React.FC<NavigationProps> = ({ onSelect }) => {
  return (
    <nav className="p-4">
      <h1 className="text-xl font-bold mb-4">The Enabler</h1>
      <ul className="space-y-2">
        <li>
          <div
            onClick={() => onSelect('query')}
            className="cursor-pointer hover:text-blue-300"
          >
            Query
          </div>
        </li>
        <li>
          <div
            onClick={() => onSelect('admin')}
            className="cursor-pointer hover:text-blue-300"
          >
            Admin
          </div>
        </li>
        <li>
          <div
            onClick={() => onSelect('shell')}
            className="cursor-pointer hover:text-blue-300"
          >
            Shell
          </div>
        </li>
      </ul>
    </nav>
  );
};

export default Navigation;
