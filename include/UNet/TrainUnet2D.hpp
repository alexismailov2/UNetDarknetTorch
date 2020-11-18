#pragma once

#include <map>
#include <string>
#include <vector>

void runOpts(std::map<std::string, std::vector<std::string>> params);
auto ParseOptions(int argc, char *argv[]) -> std::map<std::string, std::vector<std::string>>;