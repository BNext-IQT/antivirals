FROM inqtel/platform:latest
LABEL maintainer="JJ Ben-Joseph (jbenjoseph@iqt.org)" \
      description="A antivirals agent container optimised for performance and minimal attack surface."
ARG DEBIAN_FRONTEND=noninteractive
ENTRYPOINT [ "antivirals" ] 
CMD [ "up" ]
COPY setup.py README.rst /app/
COPY antivirals /app/antivirals
WORKDIR /app
RUN CFLAGS="-g0 -O3 -Wl,--strip-all -I/usr/include:/usr/local/include -L/usr/lib:/usr/local/lib" \
    pip3 install --compile --no-cache-dir --global-option=build_ext \
       --global-option="-j 4" -e .[optim] \
 && apt-get remove -y python3-dev python3-pip build-essential cmake \
      libopenblas-openmp-dev gfortran libffi-dev \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* /tmp/*