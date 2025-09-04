#include <memory>

template<typename T>
struct Node {
    T data;
    std::shared_ptr<Node<T>> next;

    Node(const T& data) : data(data), next() {}
};

template<typename T>
class LinkedList {
public:
    LinkedList() : head(nullptr) {}

    void add(const T& data) {
        auto newNode = std::make_shared<Node<T>>(data);
        if (tail) {
            tail->next = newNode;
        } else {
            head = newNode;
        }
        tail = newNode;
    }

private:
    std::shared_ptr<Node<T>> head;
    std::shared_ptr<Node<T>> tail;
};


void CreateLL()
{
    LinkedList<int> ll;
    ll.add(1);
}

int main()
{
    
    CreateLL();
    
    while(1) {}
}