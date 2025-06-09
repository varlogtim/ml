const Navigation: React.FC = () => {
  return (
    <nav className="p-4">
      <h1 className="text-2xl font-bold mb-8">The Enabler</h1>
      <ul className="space-y-2">
        <li><a href="#" className="hover:text-blue-300">Home</a></li>
        <li><a href="#" className="hover:text-blue-300">Logout</a></li>
      </ul>
    </nav>
  );
};

export default Navigation;
