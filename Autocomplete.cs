using System;
using System.Collections.Generic;
using System.Linq;

namespace Autocomplete
{
    public class TrieNode 
    {
        public char Value { get; set; }
        public Dictionary<char, TrieNode> Children { get; set; }
        public bool IsEndOfWord { get; set; }
        
        public TrieNode(char value)
        {
            Value = value;
            Children = new Dictionary<char, TrieNode>();
        }
    }

    public class Trie 
    {
        private readonly TrieNode _root;

        public Trie()
        {
            _root = new TrieNode('\0');
        }

        public void AddWord(string word)
        {
            var node = _root;
            foreach (var c in word)
            {
                if (!node.Children.ContainsKey(c))
                {
                    node.Children[c] = new TrieNode(c);
                }
                node = node.Children[c];
            }
            node.IsEndOfWord = true;
        }

        public IEnumerable<string> GetSuggestions(string prefix)
        {
            var node = _root;
            foreach (var c in prefix)
            {
                if (!node.Children.ContainsKey(c))
                {
                    return Enumerable.Empty<string>();
                }
                node = node.Children[c];
            }
            
            return GetWords(node, prefix);
        }

        private IEnumerable<string> GetWords(TrieNode node, string prefix)
        {
            if (node.IsEndOfWord)
            {
                yield return prefix;
            }
            
            foreach (var child in node.Children)
            {
                yield return string.Join("", GetWords(child.Value, $"{prefix}{child.Key}"));
            }
        }
    }
}

public class HelloWorld {
  static void Main() {
      var trie = new Autocomplete.Trie();
trie.AddWord("apple");
trie.AddWord("banana");
trie.AddWord("cherry");
trie.AddWord("apricot");
trie.AddWord("classic");
trie.AddWord("query");


var suggestions = trie.GetSuggestions("cl");
 Console.WriteLine(string.Join(", ", suggestions)); // Convert IEnumerable to string
   
  }
}
