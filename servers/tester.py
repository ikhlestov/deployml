import time

import aiohttp
import asyncio


N_STEPS = 20


async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()


async def main():
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        tasks = [fetch(session, 'http://0.0.0.0:8080/') for i in range(N_STEPS)]
        await asyncio.wait(tasks)
        time_cons = time.time() - start_time
        print("Time cons: %.2f" % time_cons)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
