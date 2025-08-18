#include <iostream>
#include <cpr/cpr.h>
#include <curl/curl.h>
#include <boost/json.hpp>




int main()
{
    boost::json::object obj;
    obj["msg"] = "Hello";
    obj["user"] = "Sebastian";
    obj["id"] = 123;

    std::string json_str = boost::json::serialize(obj);

    auto future = cpr::PostAsync(
        cpr::Url{ "https://httpbin.org/post" },
        cpr::Body{ json_str },
        cpr::Header{ {"Content-Type", "application/json"} }
    );

    // Odbierz wynik (blokuje w tym miejscu)
    cpr::Response r = future.get();

    std::cout << "Status: " << r.status_code << "\n";
    std::cout << "OdpowiedŸ:\n" << r.text << "\n";

    return 0;
}

